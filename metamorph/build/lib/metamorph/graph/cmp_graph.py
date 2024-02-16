from __future__ import annotations
from typing import List, Union, Dict, Tuple

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from collections import OrderedDict
import transformers

import logging

from metamorph.config.config import get_log_file_loc
from metamorph.misc.types import GraphNode
from metamorph.graph.abs_graph import Graph


class Squeeze_before_Linear(nn.Module):
    def forward(self, x):
        return x[:, 0, :] if len(x.shape)==3 else x
    
class Tuple2Tensor(nn.Module):
    def forward(self, x):
        return x[0] if isinstance(x, Tuple) else x

class ComputeNode(nn.Module):
    def __init__(self, op: GraphNode, op_index=None):
        super(ComputeNode, self).__init__()
        self.input = 0
        self.next = []
        self.op = op
        self.op_index = op_index

        self.requires_flatten = False
        self.requires_grad = False
        self.requires_squeeze = False

    def add_next(self, cmp_node: ComputeNode):
        self.next.append(cmp_node)

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        self.op.requires_grad_(self.requires_grad)
    
    def requires_flatten_(self, requires_flatten=True):
        self.requires_flatten = requires_flatten
        
    def requires_squeeze_(self, requires_squeeze=True):
        self.requires_squeeze = requires_squeeze

    def forward(self, x_in: torch.Tensor):
        out = None
        if self.op == 'placeholder':
            out = x_in
        elif self.requires_flatten:
            out = self.op(torch.flatten(x_in, start_dim=1))
        elif self.requires_squeeze:
            out = self.op(x_in[:, 0, :]) if len(x_in.shape)==3 else self.op(x_in) # for transformer, specifically, ViT
        else:
            out = self.op(x_in)
        if isinstance(out, Tuple):
            return out[0]
        else:
            return out

class ComputeGraph(nn.Module):
    def __init__(self, graph: Graph, original_models: Union[List[nn.Module], Dict[Tuple[int, int], nn.Module]], load_weight: bool, use_transformer=False, device='cuda'):
        super(ComputeGraph, self).__init__()

        logging.basicConfig(filename=get_log_file_loc(), format='%(asctime)s %(levelname)-4s %(message)s',
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        if isinstance(original_models, Dict):
            logging.info('Load parameters from pre-trained Compute Graph')
        else:
            logging.info('Load Pamameters from Original Models')

        self.mod = ComputeNode('placeholder')
        self.params = []
        self.out_seq = []
        self.device = device
        self.model_offset = 1
        self.use_transformer = use_transformer

        stack = [graph.root]
        stack_op = [self.mod]
        while stack:
            for _ in range(len(stack)):
                cur_node = stack.pop(0)
                cur_op = stack_op.pop(0)
                if cur_node.children:
                    stack.extend(cur_node.children)
                    for node in cur_node.children:
                        if node.op_type == 'InsertNode':
                            # choose the type of insert_op given input and output size
                            # (h,w) dim is fixed, relax on c-dim
                            if node.input_size[2] == node.output_size[2] and node.input_size[3] == node.output_size[3]:
                                insert_op = TransformC(node.input_size, node.output_size)
                            # c-dim is fixed, relax on (h,w)-dim
                            elif node.input_size[1] == node.output_size[1]:
                                insert_op = TransformHW(node.input_size, node.output_size)
                            else: # relax on all-dim
                                insert_op = TransformCHW(node.input_size, node.output_size)
                            self.params.extend(insert_op.parameters())
                            tmp_op = ComputeNode(insert_op.to(self.device), node.op_index)
                            tmp_op.requires_grad_(node.requires_grad)
                        else:
                            tmp_op = self.load_op(node, original_models, load_weight)
                            # tmp_op.requires_grad_()  # fine tune entire graph
                        cur_op.add_next(tmp_op)
                    stack_op.extend(cur_op.next)
                else:
                    self.out_seq.append(cur_node.op_index[0])
        self.cmp_graph_sequential()

    def __str__(self) -> str:
        stack = [self.mod]
        ret = ''
        while stack:
            for _ in range(len(stack)):
                cur_op = stack.pop(0)
                ret += str(cur_op) + '\t'
                if cur_op.next:
                    stack.extend(cur_op.next)
            ret += '\n'
        return ret

    def load_op(self, node, original_models: Union[List[nn.Module], Dict[Tuple[int, int], nn.Module]], load_weight: bool) -> ComputeNode:
        if isinstance(original_models, Dict):
            op = copy.deepcopy(
                original_models[node.op_index]
            )
        else:
            op = copy.deepcopy(
                original_models[node.op_index[0]-self.model_offset][node.op_index[1]]
            )
        if not load_weight:
            if node.is_merged:
                self.init_weight(op)  # initialize the weight of conv2d, linear, and batchnorm
        self.params.extend(op.parameters())
        cmp_node = ComputeNode(op.to(self.device), node.op_index)
        cmp_node.requires_flatten_(node.requires_flatten)
        cmp_node.requires_squeeze_(node.requires_squeeze)
        cmp_node.requires_grad_(node.requires_grad)  # sub-graph fine tuning
        return cmp_node

    def init_weight(self, op: nn.Module) -> None:
        op_children = [tmp for tmp in op.children()]
        if len(op_children) == 0:
            if isinstance(op, nn.Conv2d) or isinstance(op, nn.Linear):
                nn.init.kaiming_normal_(op.weight)
            elif isinstance(op, nn.BatchNorm2d) or isinstance(op, nn.BatchNorm1d):
                op.reset_parameters()
        else:
            for tmp in op_children:
                self.init_weight(tmp)

    def train(self, mode=True) -> ComputeGraph:
        self.training = mode
        stack = [self.mod]
        while stack:
            for _ in range(len(stack)):
                cur_node = stack.pop(0)
                if cur_node.op != 'placeholder':
                    cur_node.train(mode)
                if cur_node.next:
                    stack.extend(cur_node.next)
        return self

    def eval(self) -> ComputeGraph:
        return self.train(False)

    def to(self, *args, **kwargs) -> ComputeGraph:
        device = args[0]
        if isinstance(device, str):
            stack = [self.mod]
            while stack:
                for _ in range(len(stack)):
                    cur_node = stack.pop(0)
                    if cur_node.op != 'placeholder':
                        cur_node = cur_node.to(device)
                    if cur_node.next:
                        stack.extend(cur_node.next)
            return self
        else:
            super(ComputeGraph, self).to(*args, **kwargs)

    def export_parameters(self) -> Dict[Tuple[int, int], nn.Module]:
        models_dict = dict()
        stack = [self.mod]
        while stack:
            for _ in range(len(stack)):
                cur_node = stack.pop(0)
                if cur_node.op != 'placeholder':
                    models_dict[cur_node.op_index] = cur_node.op.cpu()
                if cur_node.next:
                    stack.extend(cur_node.next)
        return models_dict

    # freeze the model, model not able to backpropagate anymore
    def freeze_all_node(self) -> None:
        stack = [self.mod]
        while stack:
            for _ in range(len(stack)):
                cur_op = stack.pop(0)
                if cur_op.requires_grad:
                    cur_op.requires_grad_(False)
                if cur_op.next:
                    stack.extend(cur_op.next)

    def check_requires_grad(self) -> None:
        stack = [self.mod]
        grad_info = ''
        req_grad_count = 0
        while stack:
            for _ in range(len(stack)):
                cur_op = stack.pop(0)
                # if cur_op.op != 'placeholder':
                #     for m in cur_op.op.modules():
                #         try:
                #             logging.info(f'Operator {m} Requires_grad {m.weight.requires_grad}')
                #         except AttributeError:
                #             continue
                if cur_op.requires_grad:
                    grad_info += f'{cur_op.op_index} '
                    req_grad_count += 1
                if cur_op.next:
                    stack.extend(cur_op.next)
        logging.info(f'Node requires grad {req_grad_count}: {grad_info}')

    def cmp_graph_sequential(self):
        li, li_sub = [], []
        li_in, c_in = [-1], [-1]
        c_child = []
        stack = [self.mod]
        self.out_seq_idx, self.out_op_idx = [], []
        while stack:
            cur_node = stack.pop()
            if cur_node.op == 'placeholder':
                li_sub.append(cur_node)
            else:
                if self.use_transformer:
                    li_sub.append(Tuple2Tensor())
                if cur_node.requires_flatten:
                    li_sub.append(nn.Flatten(start_dim=1))
                if cur_node.requires_squeeze:
                    li_sub.append(Squeeze_before_Linear())
                li_sub.append(cur_node.op)
            if cur_node.next:
                stack.extend(cur_node.next)
                if len(cur_node.next) > 1:
                    c_child.append(len(cur_node.next))
                    li_sub = nn.Sequential(*li_sub)
                    li.append(li_sub)
                    li_sub = []
                    c_in.append(len(li)-1)
                    li_in.append(c_in[-1])
            else:
                li_sub = nn.Sequential(*li_sub)
                li.append(li_sub)
                li_sub = []
                if c_child:
                    c_child[-1] -= 1
                    if c_child[-1] == 0:
                        c_in.pop()
                        c_child.pop()
                li_in.append(c_in[-1])
                self.out_seq_idx.append(len(li)-1)
                self.out_op_idx.append(cur_node.op_index[0])
        li_in.pop()
        self.nnSeq = nn.Sequential(*li)
        self.nnSeq_in = li_in

    def forward(self, x_in: torch.Tensor) -> List[torch.Tensor]:
        # final_result = [0 for _ in self.out_idx]
        final_result = []
        inter_result = []

        # hard code here for BERT
        if isinstance(x_in, Dict):
            attention_mask = x_in["attention_mask"][:,None, None,:] # extend attention mask
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
            for i, seq in enumerate(self.nnSeq):
                if self.nnSeq_in[i] == -1:
                    out = x_in["input_ids"]
                else:
                    out = inter_result[self.nnSeq_in[i]]
                for layer in self.nnSeq[i]:
                    if isinstance(layer, transformers.models.bert.modeling_bert.BertEmbeddings):
                        out = layer(out, x_in["token_type_ids"])
                    elif isinstance(layer, transformers.models.bert.modeling_bert.BertLayer):
                        out = layer(out, attention_mask)
                    else:
                        out = layer(out)
                inter_result.append(out)
                if i in self.out_seq_idx:
                    final_result.append(inter_result[-1])
        else:
            for i, seq in enumerate(self.nnSeq):
                if self.nnSeq_in[i] == -1:
                    inter_result.append(self.nnSeq[i](x_in))
                else:
                    inter_result.append(self.nnSeq[i](inter_result[self.nnSeq_in[i]]))
                if i in self.out_seq_idx:
                    final_result.append(inter_result[-1])
        # for i in range(len(self.out_seq)):
        #     final_result[self.out_seq[i]-1] = inter_result[-i-1]
        # return final_result
        return [i for _,i in sorted(zip(self.out_op_idx, final_result))]


    def forward2(self, x_in: torch.Tensor) -> List[torch.Tensor]:
        stack = [self.mod]
        inter_result = [x_in]
        final_result = []
        while stack:
            tmp_result = []
            for i in range(len(stack)):
                cur_op = stack.pop(0)
                tmp_result.append(cur_op(inter_result[cur_op.input]))
                if cur_op.next:
                    for j in cur_op.next:
                        j.input = i
                    stack.extend(cur_op.next)
                else:
                    final_result.append(tmp_result[-1])
            inter_result = tmp_result
        final_result.extend(inter_result)
        return [i for _,i in sorted(zip(self.out_seq, final_result))]


# interpolation on (h,w) dim
class TransformHW(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformHW, self).__init__()
        self.input_size = input_size
        if len(output_size) == 4:
            self.output_size = (output_size[2], output_size[3])
        else:
            self.output_size = (output_size[2],)
    
    def forward(self, x):
        if len(self.output_size) > 1:
            return F.interpolate(x, size=self.output_size, mode='bilinear')
        else:
            return F.interpolate(x, size=self.output_size, mode='linear')

# 1x1 conv to relax on C dim
class TransformC(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformC, self).__init__()
        self.trans = nn.Conv2d(input_size[1], output_size[1], kernel_size=1)

    def forward(self, x):
        return self.trans(x)

# Combining the above two, to relx on (c,h,w) dims
class TransformCHW(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformCHW, self).__init__()
        if len(output_size) == 4:
            self.output_size = (output_size[2], output_size[3])
        else:
            self.output_size = (output_size[2],)
        self.trans = nn.Conv2d(input_size[1], output_size[1], kernel_size=1)

    def forward(self, x):
        if len(self.output_size) > 1:
            return F.interpolate(self.trans(x), size=self.output_size, mode='bilinear')
        else:
            return F.interpolate(self.trans(x), size=self.output_size, mode='linear')


# matrix multiplication to relax on (h,w) dim
# X2 = W1X1W2 : (h2, w2) = (h2, h1) * (h1, w1) * (w1, w2)
class TransformHW_MM(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformHW_MM, self).__init__()
        h2, w2 = output_size[2], output_size[3]
        h1, w1 = input_size[2], input_size[3]
        self.W1 = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(h2, h1)))
        self.W2 = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(w1, w2)))

    def forward(self, x):
        return torch.matmul(torch.matmul(self.W1, x), self.W2)

# matrix multiplication to relax on (h,w) dim
# reshape input s.t. X1.shape = (N, C, h1*w2), W.shape = (h1*w1, h2*w2)
# X2 = X1W, then reshape X2
class TransformHW_Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformHW_Linear, self).__init__()
        flatten_ipt_size = input_size[2] * input_size[3]
        flatten_opt_size = output_size[2] * output_size[3]
        self.trans = nn.Sequential(
            nn.Flatten(start_dim=2, end_dim=3),
            nn.Linear(flatten_ipt_size, flatten_opt_size)
        )
        self.output_size = output_size

    def forward(self, x):
        return self.trans(x).view(self.output_size)
