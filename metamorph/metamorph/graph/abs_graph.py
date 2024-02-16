
from __future__ import annotations
from typing import List, Optional, Tuple, Union, DefaultDict

from itertools import combinations
from dataclasses import dataclass
from copy import copy, deepcopy
from collections import defaultdict, OrderedDict

import torch
import numpy as np
import torch.nn as nn
import math

import logging
from metamorph.config.config import get_log_file_loc
from metamorph.misc.types import Direction, Relation, FinetuneLevel

InputSig = torch.Size
NodeIndex = Tuple[int, int]
GraphHash = int
TopoHash = int

@dataclass
class TaskCapacity:
    task_id: int
    total: int
    shared: int
    individual: int
    # [ ((task_ids), segment_capacity) ]
    details: List[Tuple[Tuple[int], int]]

    def __str__(self) -> str:
        tid = f'Task_id: {self.task_id}\n'
        overview = f'Total: {self.total} (individual: {self.individual}, shared: {self.shared})\n'
        details = f'Details:\n'
        for segment in self.details[::-1]:
            details += f'\t{segment[0]}: {segment[1]}\n'
        ret = tid + overview + details
        return ret


@dataclass
class MergeConfig:
    input_sig: InputSig
    node1_idx: NodeIndex
    node2_idx: NodeIndex
    direction: Direction
    relation: Relation
    graph_type: str = ''
    graph_idx: int = 0


class Node:
    MAX_N_MODELS = 4    # 4-bit, 16 models maximum
    MAX_N_LAYERS = 9    # 9-bit, 512 layers maximum

    def __init__(self, op_idx=None, operator=None) -> None:
        self.input_size = 0
        self.output_size = 0
        self.op_index = (0,0)
        self.op_type = None
        self.op_desc = None
        self.capacity  = 0
        self.parent = None
        # For subgraph finetuning
        self.requires_grad = False
        # For basic flatten support
        self.requires_flatten = False
        # For transformer classifier
        self.requires_squeeze = False
        # For weight initialization
        self.is_merged = False
        self.children = []
        if op_idx and operator:
            self.locate(op_idx, operator)

    def __str__(self) -> str:
        if isinstance(self.op_type, str):
            if self.op_type == 'placeholder':
                return f"({self.op_index}){self.op_type}"
            else:
                return f"({self.parent.op_index}->{self.op_index}){self.op_type}"
        else:
            return f"({self.parent.op_index}->{self.op_index}){self.op_type.__name__}"

    def __key_loc(self) -> str:
        if isinstance(self.op_type, str):
            return "".join(self.encode_location()) + "".join(self.encode_location())
        else:
            return "".join(self.parent.encode_location()) + "".join(self.encode_location())
    
    def __key_topo(self) -> str:
        if isinstance(self.op_type, str):
            return self.op_desc
        else:
            return str(self.parent.op_desc) + str(self.op_desc)

    def __hash__(self) -> int:
        return hash(self.__key_loc())

    def topology_hash(self) -> int:
        return hash(self.__key_topo())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return self.__key_loc() == other.__key_loc()
        return NotImplemented
    
    def __copy__(self) -> Node:
        """Return a copy of this node only (without its children)."""
        node_cls = self.__class__
        new_node = node_cls.__new__(node_cls)
        new_node.__dict__.update(self.__dict__)
        new_node.children = []
        new_node.parent = None
        new_node.requires_grad = False
        new_node.is_merged = False
        return new_node

    def __deepcopy__(self, memo) -> Node:
        """Return a deep copy of this node and its children."""
        node_cls = self.__class__
        new_node = node_cls.__new__(node_cls)
        memo[id(self)] = new_node
        for k, v in self.__dict__.items():
            if k == 'children':
                setattr(new_node, k, [])
            elif k == 'parent':
                setattr(new_node, k, None)
            elif k in ['input_size', 'output_size']:
                if isinstance(v, torch.Tensor):
                    setattr(new_node, k, v.detach.clone())
                else:
                    setattr(new_node, k, v)
            else:
                setattr(new_node, k, deepcopy(v, memo))
        new_node.requires_grad = False
        new_node.is_merged = False
        return new_node

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
    
    def requires_flatten_(self, requires_flatten=True):
        self.requires_flatten = requires_flatten
        
    def requires_squeeze_(self, requires_squeeze=True):
        self.requires_squeeze = requires_squeeze

    def locate(self, op_idx: NodeIndex, operator: Union[nn.Module, str]) -> None:
        self.op_index = op_idx
        if isinstance(operator, nn.Module):
            self.op_type = type(operator)
            self.op_desc = str(operator)
            self.capacity = sum(param.numel() for param in operator.parameters())
        else:
            self.op_type = operator
            self.op_desc = 'placeholder'
            self.capacity = 0

    def encode_location(self) -> str:
        branch_idx = format(self.op_index[0], f'0{self.MAX_N_MODELS}b')[-self.MAX_N_MODELS:]
        layer_idx = format(self.op_index[1], f'0{self.MAX_N_LAYERS}b')[-self.MAX_N_LAYERS:]
        return branch_idx, layer_idx
    
    def set_io_sizes(self, input_size: torch.Tensor, output_size: torch.Tensor) -> None:
        self.input_size = input_size
        self.output_size = output_size

    def input_signature(self) -> InputSig:
        return self.input_size


class Graph:
    UNMERGEABLE_NODES = [
        nn.Sequential,
        nn.ReLU,
        nn.BatchNorm2d,
        nn.BatchNorm1d,
        nn.LayerNorm,
        # nn.Linear,
        nn.Dropout,
        nn.MaxPool2d,
        nn.AdaptiveAvgPool2d,
        nn.Flatten,
        'placeholder',
        'InsertNode'
    ]

    def __init__(self, sample_input: torch.Tensor, models: List, use_transformer=False, device='cuda'):
        logging.basicConfig(filename=get_log_file_loc(), format='%(asctime)s %(levelname)-4s %(message)s',
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

        self.sample_input = sample_input
        self.device = device
        self.model_offset = 1
        self.req_grad_nodes = FinetuneLevel.SUBGRAPH
        self.use_transformer = use_transformer

        self.root = Node((0,0), 'placeholder')
        self.root.set_io_sizes(self.sample_input.shape, self.sample_input.shape)
        self.mergeable_nodes: DefaultDict[InputSig, List[Node]] = defaultdict(list)
        self.relaxed_mergeable_nodes = None # fixed h,w channel, relax on c channel: 1st option
        self.n_models = len(models)
        self.last_op_idx = {k:0 for k in range(1, self.n_models+1)}

        for i, model in enumerate(models):
            if self.use_transformer:
                self.add_model_transformer(i+self.model_offset, model)
            else:
                self.add_model(i+self.model_offset, model)
        self.build_mergeable_nodes()

    def __str__(self) -> str:
        stack = [self.root]
        ret = ''
        while stack:
            for _ in range(len(stack)):
                cur_node = stack.pop(0)
                ret += str(cur_node) + '\t'
                if cur_node.children:
                    stack.extend(cur_node.children)
            ret += '\n'
        return ret

    def __key_loc(self) -> str:
        k = []
        stack = [self.root]
        while stack:
            stack.sort(key=lambda x: x.op_index)
            k.extend([hash(node) for node in stack])
            for _ in range(len(stack)):
                cur_node = stack.pop(0)
                if cur_node.children:
                    stack.extend(cur_node.children)
        return tuple(k)

    def __key_topo(self) -> str:
        k = []
        stack = [self.root]
        while stack:
            stack.sort(key=lambda x: x.op_index)
            k.extend([node.topology_hash() for node in stack])
            for _ in range(len(stack)):
                cur_node = stack.pop(0)
                if cur_node.children:
                    stack.extend(cur_node.children)
        return tuple(k)

    def __hash__(self):
        return hash(self.__key_loc())

    def topology_hash(self) -> int:
        return hash(self.__key_topo())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Graph):
            return self.__key_loc() == other.__key_loc()
        return NotImplemented

    def __copy__(self) -> Graph:
        new_graph = Graph(
            sample_input=self.sample_input.detach().clone(),
            models=[],
            use_transformer=self.use_transformer, 
            device=self.device
        )
        new_root = new_graph.root
        new_stack = [new_root]
        stack = [self.root]
        while stack:
            for _ in range(len(stack)):
                cur_node = new_stack.pop(0)
                node = stack.pop(0)
                if node.children:
                    stack.extend(node.children)
                    for child in node.children:
                        new_child = deepcopy(child)
                        new_child.parent = cur_node
                        cur_node.children.append(new_child)
                    new_stack.extend(cur_node.children)
        new_graph.last_op_idx = self.last_op_idx
        new_graph.build_mergeable_nodes()
        return new_graph
    
    def __deepcopy__(self, memo) -> Graph:
        return self.__copy__()

    def add_model(self, mod_idx: int, model: List[nn.Module]) -> None:
        cur_node = self.root
        cur_output = self.sample_input
        cur_idx = 0
        for layer in model:
            new_node = Node((mod_idx, cur_idx), layer)
            # compute input and output size
            input_size = cur_output.shape
            try:
                cur_output = layer(cur_output) # update the current output
            except RuntimeError:
                print(f"Encountered an error at {new_node.op_index}. Try to insert a nn.Flatten layer ... ", end = "")
                flatten = nn.Flatten()
                cur_output = layer(flatten(cur_output))
                print("Success!")
                new_node.requires_flatten_()
            output_size = cur_output.shape
            new_node.set_io_sizes(input_size, output_size)
            # add children and parent
            cur_node.children.append(new_node)
            new_node.parent = cur_node
            cur_node = new_node
            cur_idx += 1
        self.last_op_idx[mod_idx] = cur_idx - 1
        
    def add_model_transformer(self, mod_idx: int, model: List[nn.Module]) -> None:
        cur_node = self.root
        cur_output = self.sample_input
        cur_idx = 0
        for layer in model:
            new_node = Node((mod_idx, cur_idx), layer)
            # compute input and output size
            input_size = cur_output.shape
            try:
                cur_output = layer(cur_output) # update the current output
                if isinstance(cur_output, Tuple):
                    cur_output = cur_output[0]
            except RuntimeError:
                print(f"Encountered an error at {new_node.op_index}. Try to insert a nn.Flatten layer ... ", end = "")
                flatten = nn.Flatten()
                cur_output = layer(flatten(cur_output))
                print("Success!")
                new_node.requires_flatten_()
            if isinstance(layer, nn.Linear):
                new_node.requires_squeeze_()
            output_size = cur_output.shape
            new_node.set_io_sizes(input_size, output_size)
            # add children and parent
            cur_node.children.append(new_node)
            new_node.parent = cur_node
            cur_node = new_node
            cur_idx += 1
        self.last_op_idx[mod_idx] = cur_idx - 1

    def build_mergeable_nodes(self) -> None:
        self.mergeable_nodes = defaultdict(list)
        self.relaxed_mergeable_nodes = None
        stack = [self.root]
        while stack:
            cur_node = stack.pop(0)
            if cur_node.children:
                stack.extend(cur_node.children)
            if cur_node.op_type not in Graph.UNMERGEABLE_NODES:
                key = cur_node.input_signature()
                self.mergeable_nodes[key].append(cur_node)
        if self.use_transformer:
            avail_relax_dim = ['hw']
        else:
            avail_relax_dim = ['hw', 'c']
        for relax_dim in avail_relax_dim: # add 'chw'
            self.build_relaxed_mergeable_nodes(relax_dim)

    def build_relaxed_mergeable_nodes(self, relax_dim='c') -> None:
        num_mergeable_nodes = np.sum([math.comb(len(self.mergeable_nodes[i]), 2) for i in self.mergeable_nodes])
        if num_mergeable_nodes >= 200 and not self.use_transformer:
            return
        else: # if there is not enough mergeable nodes, relax the constraint: 
            # two nodes are mergeable given the relaxed rules
            # if self.relaxed_mergeable_nodes:
            #     num_relax_comb = np.sum([math.comb(len(self.relaxed_mergeable_nodes[i]), 2) for i in self.relaxed_mergeable_nodes])
            #     # if already have enough candidates by relaxing
            #     if num_relax_comb >= 200:
            #         return
            if not self.relaxed_mergeable_nodes:
                self.relaxed_mergeable_nodes = defaultdict(list)
            stack = [self.root]
            while stack:
                cur_node = stack.pop(0)
                if cur_node.children:
                    stack.extend(cur_node.children)
                if cur_node.op_type not in Graph.UNMERGEABLE_NODES:
                    if len(cur_node.input_signature()) != 4 and not self.use_transformer: # OP After flatten, e.g., Linear
                        key = cur_node.input_signature()
                        if cur_node in self.relaxed_mergeable_nodes[key]:
                            continue
                    elif relax_dim == 'c':
                        # the c dim is relaxable, the (h,w) dim is fixed which is the key
                        key = (cur_node.input_signature()[2], cur_node.input_signature()[3]) #(h,w)-dim
                    elif relax_dim == 'hw':
                        # the (h,w) dim is relaxable, the c dim is fixed which is the key
                        key = (cur_node.input_signature()[1]) # the channel dimention
                    else: # relax_dim =='chw'
                        # all the dims are relaxable
                        key = 'all'
                        # pass
                    self.relaxed_mergeable_nodes[key].append(cur_node)
    
    def capacity(self) -> Tuple[int, List[TaskCapacity]]:
        def capacity_inner(root: Node, cur_capacity: int) -> Tuple[int, List[TaskCapacity]]:
            cur_node = root
            new_capacity = 0
            total_capacity = 0
            capacities = []
            cur_id = cur_node.op_index[0]
            shared_ids = set([cur_id]) if cur_id != 0 else set()

            while cur_node.children:
                new_capacity += cur_node.capacity
                if len(cur_node.children) == 1:
                    cur_node = cur_node.children[0]
                else:
                    for child in cur_node.children:
                        sum_branch_capacities, task_capacities = capacity_inner(child, cur_capacity+new_capacity)
                        capacities.extend(task_capacities)
                        total_capacity += sum_branch_capacities
                    for task_capacity in capacities:
                        task_ids = task_capacity.details[-1][0]
                        shared_ids.update(task_ids)
                    for task_capacity in capacities:
                        task_capacity.details.append((tuple(shared_ids), new_capacity))
                    new_capacity += total_capacity
                    return new_capacity, capacities

            model_idx = cur_node.op_index[0]
            new_capacity += cur_node.capacity
            total_capacity = cur_capacity + new_capacity
            task_capacity = TaskCapacity(
                task_id=model_idx,
                total=total_capacity,
                shared=cur_capacity,
                individual=new_capacity,
                details=[((model_idx,), new_capacity)]
            )
            return new_capacity, [task_capacity]

        total_capacity, model_capacities = capacity_inner(self.root, 0)
        model_capacities.sort(key=lambda x: x.task_id)
        return total_capacity, model_capacities
    
    def export_capacity_numpy(self) -> np.ndarray:
        graph_capacity = self.capacity()
        cur_cap = [graph_capacity[0]]
        task_list = [i for i in range(1, len(graph_capacity[1])+1)]
        task_share_pattern = []
        for i in range(1, len(graph_capacity[1])+1):
            task_share_pattern.extend(list(combinations(task_list, i)))

        tmp = OrderedDict(zip(task_share_pattern, [0 for _ in range(len(task_share_pattern))]))
        for task_cap in graph_capacity[1]:
            tmp.update(dict(task_cap.details))
            cur_cap.append(task_cap.total)
        cur_cap.extend(tmp.values())
        return np.array(cur_cap)

    def merge_sanity_check(self, num_models, level=1) -> None:
        keys = [key for key in self.mergeable_nodes.keys() if len(self.mergeable_nodes[key]) >= num_models]
        if len(keys) == 0:
            return 
        if level == 1:
            key = 1
        elif level == 2:
            key = len(keys) // 3
        elif level == 3:
            key = len(keys) // 2
        else:
            key = len(keys) - 1
        nodes = self.mergeable_nodes[keys[key]]
        n_nodes = len(nodes)
        exist_op_index = [0]
        for n in range(num_models-1):
            for i in range(n_nodes):
                np.random.seed(i+1)
                node1_idx, node2_idx = np.random.choice(range(n_nodes), size=2, replace=False)
                node1, node2 = nodes[node1_idx], nodes[node2_idx]
                if node1.parent != node2.parent and node1.op_index[0] != node2.op_index[0]:
                    if node1.op_index[0] not in exist_op_index or node2.op_index[0] not in exist_op_index:
                        break
            condition1 = ((node1.op_index[1] + node2.op_index[1]) != 0)
            if condition1:
                direction, relation = self.infer_direction_relation(node1, node2)
                if direction == Direction.NOT_VALID:
                    print("Not able to merge!")
                    return
                else:
                    merge_info = f"Merging {direction}:\n    Node1: {node1}\n    Node2: {node2}"
                    logging.info(merge_info)
                    self.merge_nodes(node1, node2, direction, relation)

    def force_connect(self, merge_config: MergeConfig) -> None:
        node1, node2 = None, None

        # print(merge_config.input_sig, merge_config.node1_idx, merge_config.node2_idx)
        # for i in self.mergeable_nodes[merge_config.input_sig]:
        #     print(i.op_index)
        
        for n in self.mergeable_nodes[merge_config.input_sig]:
            if n.op_index == merge_config.node1_idx:
                node1 = n
            elif n.op_index == merge_config.node2_idx:
                node2 = n
        if node1 is not None and node2 is not None:
            merge_info = f"Merging {merge_config.direction}:\n    Node1: {node1}\n    Node2: {node2}"
            logging.info(merge_info)
            self.merge_nodes(node1, node2, merge_config.direction, merge_config.relation)
        else:
            raise ValueError('Invalid nodes index')

    def manual_connect(self, merge_config: MergeConfig, verbose=False) -> Optional[MergeConfig]:
        node1, node2 = None, None
        for n in self.mergeable_nodes[merge_config.input_sig]:
            if n.op_index == merge_config.node1_idx:
                node1 = n
            elif n.op_index == merge_config.node2_idx:
                node2 = n

        if node1 is not None and node2 is not None:
            # Prevent merging the first layer with the "Input" node
            condition1 = ((node1.op_index[1] + node2.op_index[1]) != 0)
            # Prevent merging when two nodes are from the same branch
            # condition2 = node1.op_index[0] != node2.op_index[0]
            if condition1:
                if self.check_dependency(node1, node2, merge_config.direction, merge_config.relation):

                    if verbose:
                        merge_info = f"Merging {merge_config.direction}:\n    Node1: {node1}\n    Node2: {node2}"
                        # logging.info(merge_info)
                        print(merge_info)
                    
                    self.merge_nodes(node1, node2, merge_config.direction, merge_config.relation)
                    return merge_config
            # print("Invalid Merge Config")
            return MergeConfig(torch.tensor([-1]), tuple(), tuple(), Direction.NOT_VALID, Relation.NO_RELATION)
        else:
            raise ValueError('Invalid nodes index')

    def random_connect(self, n_trial=10, verbose=False) -> Optional[MergeConfig]:
        counter = 0
        while counter < n_trial:
            n_nodes = 0
            keys = [key for key in self.mergeable_nodes.keys() if len(self.mergeable_nodes[key]) > 1]

            # while n_nodes < 2 and counter < 10:
            #     rand_key = np.random.randint(len(keys))
            #     nodes = self.mergeable_nodes[keys[rand_key]]
            #     n_nodes = len(nodes)
            #     counter += 1
            # if n_nodes < 2:
            #     break
            if len(keys) == 0:
                break
            else:
                rand_key = np.random.randint(len(keys))
                nodes = self.mergeable_nodes[keys[rand_key]]
                n_nodes = len(nodes)

            node1_idx, node2_idx = np.random.choice(range(n_nodes), size=2, replace=False)
            node1, node2 = nodes[node1_idx], nodes[node2_idx]

            # Prevent merging the first layer with the "Input" node
            condition1 = ((node1.op_index[1] + node2.op_index[1]) != 0)
            # Prevent merging when two nodes are from the same branch
            # condition2 = node1.op_index[0] != node2.op_index[0]
            if condition1:
                direction, relation = self.infer_direction_relation(node1, node2)
                if direction == Direction.NOT_VALID:
                    continue
                else:
                    merge_info = f"Merging {direction}:\n    Node1: {node1}\n    Node2: {node2}"
                    logging.info(merge_info)
                    
                    if verbose:
                        print(merge_info)
                    self.merge_nodes(node1, node2, direction, relation)

                    merge_config = MergeConfig(keys[rand_key], node1.op_index, node2.op_index, direction, relation)
                    return merge_config
            counter += 1
        
        print("Timeout: No nodes can be merged.")
        return MergeConfig(torch.tensor([-1]), tuple(), tuple(), Direction.NOT_VALID, Relation.NO_RELATION)

    # add a new layer before the current node
    def add_inter_placeholder(self, cur_node:Node) -> None:
        prev_node = cur_node.parent

        # the idx of the inserted node = 1 + the idx of the last node of the model
        self.last_op_idx[cur_node.op_index[0]] += 1
        new_node_idx = (cur_node.op_index[0], self.last_op_idx[cur_node.op_index[0]])
        new_node = Node(new_node_idx, 'InsertNode')
        new_node.set_io_sizes(prev_node.output_size, cur_node.input_size)
        new_node.requires_grad_()
        # TODO: whether to calculate the capacity of the new node here

        # connect prev_node -> new_node -> cur_node
        for i, n in enumerate(prev_node.children):
            if n.op_index == cur_node.op_index:
                prev_node.children[i] = new_node
        new_node.parent = prev_node
        new_node.children.append(cur_node)
        cur_node.parent = new_node
    
    def relaxed_random_connect(self, n_trial=10, verbose=False) -> Optional[MergeConfig]:
        counter = 0
        while counter < n_trial:
            n_nodes = 0
            keys = [key for key in self.relaxed_mergeable_nodes.keys() if len(self.relaxed_mergeable_nodes[key]) > 1]
            if len(keys) == 0:
                break
            else:
                rand_key = np.random.randint(len(keys))
                nodes = self.relaxed_mergeable_nodes[keys[rand_key]]
                n_nodes = len(nodes)
            node1_idx, node2_idx = np.random.choice(range(n_nodes), size=2, replace=False)
            node1, node2 = nodes[node1_idx], nodes[node2_idx]

            # Prevent merging the first layer with the "Input" node
            condition1 = ((node1.op_index[1] + node2.op_index[1]) != 0)
            # Prevent merging when two nodes are from the same branch
            # condition2 = node1.op_index[0] != node2.op_index[0]
            if condition1:
                direction, relation = self.infer_direction_relation(node1, node2)
                if direction == Direction.NOT_VALID:
                    continue
                else:
                    merge_info = f"Merging {direction}:\n    Node1: {node1}\n    Node2: {node2}"
                    insert_loc = 'node1' if direction == Direction.RIGHT else 'node2'
                    logging.info(merge_info)
                    
                    if verbose:
                        print(merge_info)
                    self.merge_nodes(node1, node2, direction, relation)
                    # insert a new node it the input size does not match
                    if node1.input_size != node2.input_size:
                        insert_before_node = node1 if insert_loc == 'node1' else node2
                        self.add_inter_placeholder(insert_before_node)
                        insert_info = f"Insert transform at {insert_loc}, node1 {node1.input_size}, node2 {node2.input_size}"
                        logging.info(insert_info)

                    merge_config = MergeConfig(keys[rand_key], node1.op_index, node2.op_index, direction, relation)
                    return merge_config
            counter += 1
        
        print("Timeout: No nodes can be merged.")
        return MergeConfig(torch.tensor([-1]), tuple(), tuple(), Direction.NOT_VALID, Relation.NO_RELATION)

    def check_parent_child_violation(self, node1: Node, node2: Node) -> Relation:
        # Check if a parent node is trying to connect to a child node
        #
        #   1 -> 2 -> 3 -> 4
        #      \ 5 -> 6 -> 7
        #
        # We cannot merge 5 to 7 or 2 to 7 etc.

        cur_node = node2
        while cur_node != self.root:
            if cur_node == node1:
                # node1 is a parent of node2
                return Relation.PARENT_CHILD
            cur_node = cur_node.parent

        cur_node = node1
        while cur_node != self.root:
            if cur_node == node2:
                # node2 is a parent of node1
                return Relation.CHILD_PARENT
            cur_node = cur_node.parent
        
        # node1 and node2 are not related
        return Relation.NO_RELATION
    
    def check_branching_violation(self, node1: Node, node2: Node, direction: Direction) -> bool:
        # Check if a the merging node has a branching point in its parents
        #
        #
        #   1 -> 2 -> 3 -> 4 -> 5
        #      \ 6 -> 7 -> 8
        #           \ 9 -> 10
        #
        # We cannot merge 7 to 2 etc.

        if direction == Direction.LEFT:
            pass
        elif direction == Direction.RIGHT:
            node1, node2 = node2, node1
        else:
            # Never Happens 
            raise ValueError("Direction must be either 'left' or 'right'")

        cur_node = node2
        while cur_node != self.root:
            if len(cur_node.children) > 1:
                branching_node = cur_node
                branching_prev_node = branching_node
                while branching_prev_node != self.root:
                    if branching_prev_node == node1:
                        return False
                    else:
                        branching_prev_node = branching_prev_node.parent
                
                left_prev_node = node1
                while left_prev_node != self.root:
                    if left_prev_node == branching_node:
                        return False
                    else:
                        left_prev_node = left_prev_node.parent
                return True
            else:
                cur_node = cur_node.parent
        return False

    def infer_direction_relation(self, node1: Node, node2: Node) -> Union[Direction, Relation]:
        # Assume node1 is at the left`and node2 is at the right`
        # Check if left connection is valid
        relation = self.check_parent_child_violation(node1, node2)

        if self.check_dependency(node1, node2, Direction.LEFT, relation):
            return Direction.LEFT, relation
        elif self.check_dependency(node1, node2, Direction.RIGHT, relation):
            return Direction.RIGHT, relation
        else:
            return Direction.NOT_VALID, relation
    
    def check_dependency(self, node1: Node, node2: Node, direction: Direction, relation: Relation) -> bool:
        if relation == Relation.PARENT_CHILD:
            if direction == Direction.LEFT:
                return not self.check_branching_violation(node1, node2, direction)
            return False
        elif relation == Relation.CHILD_PARENT:
            if direction == Direction.RIGHT:
                return not self.check_branching_violation(node1, node2, direction)
            return False
        else:
            return not self.check_branching_violation(node1, node2, direction)

    def merge_nodes(self, node1: Node, node2: Node, direction: Direction, relation: Relation) -> None:
        if direction == Direction.LEFT:
            prev_node, cur_node = node2, node2.parent
        elif direction == Direction.RIGHT:
            prev_node, cur_node = node1, node1.parent
        else:
            raise ValueError("Invalid direction: direction must be 'left' or 'right'")

        # 1. Parent-Child Relation or Child-Parent Relation
        #    delete all parent nodes of cur_node till the first branching point
        # 2. No Relation
        #    append node2 to node1's children and reconnect node1's parent to node2
        while cur_node:
            if cur_node == self.root:
                # Reached the root without finding any branching point
                if direction == Direction.LEFT:
                    if relation == Relation.NO_RELATION:
                        cur_node.children.remove(prev_node)
                    else:
                        node1.parent.children = []
                else:
                    if relation == Relation.NO_RELATION:
                        cur_node.children.remove(prev_node)
                    else:
                        node2.parent.children = []
                break
            if len(cur_node.children) == 1:
                # if node1 and node2 are in the same branch, node1 is parent, node2 is child
                if direction == Direction.LEFT:
                    if cur_node == node1:
                        node1.parent.children.remove(node1)
                        break
                # if node1 and node2 are in the same branch, node2 is parent, node1 is child
                else:
                    if cur_node == node2:
                        node2.parent.children.remove(node2)
                        break
                # No branching point, keep tracing back to root
                prev_node, cur_node = cur_node, cur_node.parent
            else:
                # When we reach to the first branching point
                cur_node.children.remove(prev_node)
                break

        if node1 == self.root:
            node1.children.append(node2)
            node2.parent = node1
        elif node2 == self.root:
            node2.children.append(node1)
            node1.parent = node2
        elif direction == Direction.LEFT:
            node1.parent.children.append(node2)
            node2.parent = node1.parent
        elif direction == Direction.RIGHT:
            node2.parent.children.append(node1)
            node1.parent = node2.parent
        else:
            raise ValueError("direction must be 'left' or 'right'")  # should never reach here

        if direction == Direction.LEFT:
            self.mark_finetune_nodes(node1, node2)
        else:
            self.mark_finetune_nodes(node2, node1)

    def set_req_grad(self, req_grad_nodes=FinetuneLevel.SUBGRAPH):
        self.req_grad_nodes = req_grad_nodes
        
    def mark_finetune_nodes(self, node1: Node, node2 : Node) -> None:
        if self.req_grad_nodes == FinetuneLevel.SUBGRAPH:
            cur_node = node1
            while cur_node != self.root:
                cur_node.requires_grad_()
                # allow gradients update for all of its parent nodes (including the node itself)
                cur_node = cur_node.parent
            node2.requires_grad_()  
        # allow its direct children to be finetuned as well
        # for child in node1.children:
        #     child.requires_grad_()

        # for child in node2.children:
        #     child.requires_grad_()    

        elif self.req_grad_nodes == FinetuneLevel.FULLGRAPH:
            stack = [self.root]
            while stack:
                for _ in range(len(stack)):
                    cur_node = stack.pop(0)
                    cur_node.requires_grad_()
                    if cur_node.children:
                        stack.extend(cur_node.children)
        else:
            raise ValueError("req_grad_nodes must be either FinetuneLevel.FULLGRAPH or 'sub'")
        
        # mark the nodes that are merged
        cur_node = node1.parent
        while cur_node != self.root:
            cur_node.is_merged = True
            cur_node = cur_node.parent

    
    # # Overwriting methods above. Need to be commented out in GMorph.
    # # Only used for testing merging nodes with totally different input size
    # # which means (c,h,w) 3 dims are all different
    # def build_mergeable_nodes(self) -> None:
    #     self.relaxed_mergeable_nodes = None
    #     self.build_relaxed_mergeable_nodes('chw')

    # def relaxed_random_connect(self, n_trial=20, verbose=False) -> Optional[MergeConfig]:
    #     counter = 0
    #     while counter < n_trial:
    #         logging.info(counter)
    #         nodes = self.relaxed_mergeable_nodes['all']
    #         n_nodes = len(nodes)
    #         node1_idx, node2_idx = np.random.choice(range(n_nodes), size=2, replace=False)
    #         node1, node2 = nodes[node1_idx], nodes[node2_idx]
    #         if (node1.input_size[2] == node2.input_size[2] and node1.input_size[3] == node2.input_size[3]) \
    #                 or node1.input_size[1] == node2.input_size[1]:
    #             counter += 1
    #             continue
    #         # Prevent merging the first layer with the "Input" node
    #         condition1 = ((node1.op_index[1] + node2.op_index[1]) != 0)
    #         # Prevent merging when two nodes are from the same branch
    #         # condition2 = node1.op_index[0] != node2.op_index[0]
    #         if condition1:
    #             direction, relation = self.infer_direction_relation(node1, node2)
    #             if direction == Direction.NOT_VALID:
    #                 continue
    #             else:
    #                 merge_info = f"Merging {direction}:\n    Node1: {node1}\n    Node2: {node2}"
    #                 insert_loc = 'node1' if direction == Direction.RIGHT else 'node2'
    #                 logging.info(merge_info)
                    
    #                 if verbose:
    #                     print(merge_info)
    #                 self.merge_nodes(node1, node2, direction, relation)
    #                 # insert a new node it the input size does not match
    #                 insert_before_node = node1 if insert_loc == 'node1' else node2
    #                 self.add_inter_placeholder(insert_before_node)
    #                 insert_info = f"Insert transform at {insert_loc}, node1 {node1.input_size}, node2 {node2.input_size}"
    #                 logging.info(insert_info)

    #                 merge_config = MergeConfig('all', node1.op_index, node2.op_index, direction, relation)
    #                 return merge_config
    #         counter += 1
        
    #     print("Timeout: No nodes can be merged.")
    #     return MergeConfig(torch.tensor([-1]), tuple(), tuple(), Direction.NOT_VALID, Relation.NO_RELATION)
