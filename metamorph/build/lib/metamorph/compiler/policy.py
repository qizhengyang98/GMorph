import itertools
from typing import DefaultDict, Tuple, List, Dict, Callable, Optional, Union

import copy
import random
from dataclasses import asdict
from itertools import combinations
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from metamorph.graph.abs_graph import Graph, MergeConfig, Node, InputSig, NodeIndex, GraphHash, TopoHash, TaskCapacity
from metamorph.graph.cmp_graph import ComputeGraph
from metamorph.config.config import get_log_file_loc
from metamorph.metrics.testing_utils import HiddenPrints
from metamorph.misc.types import FinetuneLevel, Direction, Relation

import logging
import json

from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from math import ceil, floor, comb, sqrt


class Policy:
    """
    Abstract class for policies that optimize multi-branch models
    """

    def __init__(self):
        pass

    def step(self, cur_epoch: int) -> Optional[Tuple[Graph, torch.Tensor]]:
        """
        Make a step in the policy given the current state and actions
        """
        raise NotImplementedError

    def early_stop(self) -> bool:
        """
        Early stop the policy
        """
        raise NotImplementedError


class SimulatedAnnealingPolicy(Policy):
    """
    A policy based on simulated annealing
    """
    COOLING_SCHEDULES = (
        'linear', 'fast', 'exponential',
        'boltzmann', 'boltzExp', 'constExp'
    )

    def __init__(
        self,
        base_graph: Graph,
        models: List[torch.nn.Module],
        f_finetune: Callable,
        f_latency: Callable,
        f_accuracy: Callable,
        load_weight: bool,
        n_merge_per_epoch=2,
        accuracy_tolerence=0.02,
        n_elites=16,
        initial_temp=90,
        min_temp=1,
        alpha=0.9,
        cooling_schedule='exponential',
        cooling_threshold=20,
        fine_tune_level=FinetuneLevel.SUBGRAPH,
        use_transformer=False, 
        device='cuda'
    ):
        super().__init__()
        # Temperature
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.min_temp = min_temp
        
        # Optimization schedule
        self.base_graph = base_graph
        self.models = models
        self.p = 0
        self.n_merge_per_epoch = n_merge_per_epoch
        self.accuracy_tolerence = accuracy_tolerence
        self.n_elites = n_elites
        self.elite_candidates = []
        self.exist_graphs: Dict[GraphHash, Dict[NodeIndex, nn.Module]] = {}
        self.exist_topologies: DefaultDict[TopoHash, bool] = defaultdict(dict)
        self.device = device
        self.load_weight = load_weight
        self.use_transformer = use_transformer

        # Cooling schedule
        if cooling_schedule not in SimulatedAnnealingPolicy.COOLING_SCHEDULES:
            raise ValueError(
                f'cooling_schedule must be one of {SimulatedAnnealingPolicy.COOLING_SCHEDULES}'
            )
        self.cooling_schedule = cooling_schedule
        self.cooling_threshold = cooling_threshold
        self.cooling_mapping = {
            'linear': self.__linear_cooling,
            'fast': self.__fast_cooling,
            'exponential': self.__exponential_cooling,
            'boltzmann': self.__boltzmann_cooling,
            'boltzExp': self.__boltzExp_cooling,
            'constExp': self.__constExp_cooling
        }
        self.alpha = alpha

        # Evaluation functions
        self.f_finetune = f_finetune
        self.f_latency = f_latency
        self.f_accuracy = f_accuracy
        
        # sub-graph finetune = FinetuneLevel.SUBGRAPH, entire-graph finetune = FinetuneLevel.FULLGRAPH
        self.fine_tune_level = fine_tune_level
        
        # Baselines
        self.accuracy_baseline, self.latency_baseline = self.get_baselines()

        # log the merge history
        self.history: List[MergeConfig] = []
        self.export_history_record = False

        logging.basicConfig(filename=get_log_file_loc(), format='%(asctime)s %(levelname)-4s %(message)s',
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

        logging.info(f'Policy settings: initial_temp {self.initial_temp}, min_temp {self.min_temp}, alpha {self.alpha}')
        logging.info(f'Policy settings: accuracy_tolerence {self.accuracy_tolerence}, n_merge_per_epoch {self.n_merge_per_epoch}, cooling_schedule: {self.cooling_schedule}, cooling_threshold: {self.cooling_threshold}')
        logging.info(f'Finetune Module: {self.fine_tune_level}')
        acc_str = ''
        for i, acc in enumerate(self.accuracy_baseline):
            acc_str += f"net{i+1}: {acc.item()*100}%   "
        logging.info(f'Baselines Acc: {acc_str}')
        logging.info(f'Baseline Latency: {self.latency_baseline}')

        # Analyze
        self.analyze_accuracy_drop = []
        self.analyze_task_total_capacity = []
        self.analyze_task_specific_capacity = []

    def save_analyze(self, path: str) -> None:
        """
        Check the relationship between acc drop and capacity
        """
        self.analyze_accuracy_drop = self.analyze_accuracy_drop
        self.analyze_task_total_capacity = self.analyze_task_total_capacity
        self.analyze_task_specific_capacity = self.analyze_task_specific_capacity
        analyze_dict = {'acc_drop': self.analyze_accuracy_drop,
                        'task_total_capacity': self.analyze_task_total_capacity,
                        'task_specific_capacity': self.analyze_task_specific_capacity}
        with open(path, 'w') as f:
            json.dump(analyze_dict, f)
    
    def step(self, cur_epoch: int) -> Optional[Tuple[Graph, torch.Tensor]]:
        """
        Make a step in the policy given the current state and actions 
            using simulated annealing
        """
        print("The current number of candidates is {}, the value of P is {}".format(len(self.elite_candidates), self.p))
        logging.info(f'Current Temp: {self.current_temp}, Current probability: {self.p}')
        logging.info(f'Current number of candidates is {len(self.elite_candidates)}')
        logging.info(f'Current number of generated Graphs: {len(self.exist_graphs)}, Topologies: {len(self.exist_topologies)}')

        if not self.elite_candidates or self.p < random.uniform(0, 1):
            print("Optimizing on the Original Graph ...")
            logging.info("Optimizing on the Original Graph ...")
            cur_graph = self.get_merged_graph(self.base_graph, cur_epoch)
            loaded_models = self.models
            for i in range(self.n_merge_per_epoch):
                self.history[cur_epoch*self.n_merge_per_epoch+i].graph_type = 'original_graph'
        else:
            print("Optimizing on Elite Candidate ...") 
            logging.info("Optimizing on Elite Candidate ...")
            idx = random.randint(0, len(self.elite_candidates)-1)
            logging.info(f'Optimizing on elite_candidate {idx}')
            cur_graph = self.get_merged_graph(self.elite_candidates[idx][0], cur_epoch)
            loaded_models = self.exist_graphs[hash(self.elite_candidates[idx][0])]
            for i in range(self.n_merge_per_epoch):
                self.history[cur_epoch*self.n_merge_per_epoch+i].graph_type = 'elite_candidate'
                self.history[cur_epoch*self.n_merge_per_epoch+i].graph_idx = idx

        if self.export_history_record:
            self.export_merge_history()

        cur_graph_hash = hash(cur_graph)
        candidate, cmp_graph = None, None
        if cur_graph_hash not in self.exist_graphs.keys():
            self.exist_graphs[cur_graph_hash] = None
            
            # if current graph topo already exist, check whether it is valid (pass)
            cur_graph_topo_hash = cur_graph.topology_hash()
            if cur_graph_topo_hash in self.exist_topologies and self.exist_topologies[cur_graph_topo_hash] == 1:
                print('Graph Topology is already valid! Skip this iteration ...')
                logging.info('Graph Topology is already valid! Skip this iteration ...')
                return candidate, cmp_graph
            # if not exist, add it into self.exist_topologies
            elif cur_graph_topo_hash not in self.exist_topologies:
                self.exist_topologies[cur_graph_topo_hash] = 0
            
            logging.info(f'Current abs-Graph:\n {cur_graph}')
            total_capacity, branch_capacities = cur_graph.capacity()
            logging.info(f'total capacity: {total_capacity}')
            logging.info('task specific capacity: ')
            for i, cap in enumerate(branch_capacities):
                logging.info(cap)

            # Generate a new candidate
            cmp_graph = ComputeGraph(cur_graph, loaded_models, load_weight=self.load_weight, use_transformer=self.use_transformer, device=self.device)
            
            # logging.info(f'Current cmp-Graph:\n {cmp_graph.nnSeq}')
            # logging.info(f'forward sequence: {cmp_graph.nnSeq_in}')
            
            logging.info('Finetune start ...')
            self.f_finetune(cmp_graph, self.accuracy_baseline, self.accuracy_tolerence)
            # Compute accuracy drop
            with HiddenPrints():
                cur_accuracy = self.f_accuracy(cmp_graph)
            # get the avg accuracy drop between tasks
            # acc_delta = sum(self.accuracy_baseline - cur_accuracy) / len(self.accuracy_baseline)
            # get the max accuracy drop between tasks
            task_acc_delta = self.accuracy_baseline - cur_accuracy
            acc_delta = torch.max(task_acc_delta)
            print('Finetune stop, the accuracy drop is: ', acc_delta.item())

            # Analyze
            self.analyze_accuracy_drop.append(task_acc_delta.tolist())
            self.analyze_task_total_capacity.append([b.total for b in branch_capacities])
            self.analyze_task_specific_capacity.append([b.individual for b in branch_capacities])

            # Update next temperature
            self.current_temp = self.get_temperature(cur_epoch+1)
            # Update next probability
            self.p = self.get_probability(acc_delta)
            # Compute latency
            cur_latency = self.f_latency(cmp_graph)

            # Log accuracy and latency 
            acc_str = ''
            for i, acc in enumerate(cur_accuracy):
                acc_str += f"net{i+1}: {acc.item()*100}%   "
            logging.info(f'Current Acc: {acc_str}')
            logging.info(f'Finetune ends, the accuracy drop : {acc_delta.item()}, latency : {cur_latency}')

            # Update the candidate
            if acc_delta <= self.accuracy_tolerence:
                logging.info("Acc drop MEET the threshold, add graph to elite candidates ... ")
                self.exist_graphs[cur_graph_hash] = cmp_graph.export_parameters()
                # update current graph topo in self.exist_topologies
                self.exist_topologies[cur_graph_topo_hash] = 1

                if len(self.elite_candidates) < self.n_elites:
                    self.elite_candidates.append((cur_graph, cur_latency))
                    candidate = self.elite_candidates[-1]
                else:
                    # if the elite candidate set is full
                    logging.info("Elite candidate set is full, remove the oldest graph ...")
                    candidate = self.elite_candidates.pop(0) # remove the oldest graph
                    self.elite_candidates.append((cur_graph, cur_latency))
                    candidate = self.elite_candidates[-1]
        else:
            print('Graph exists! Skip this epoch ...')
            logging.info('Graph exists! Skip this iteration ...')
        return candidate, cmp_graph 

    def early_stop(self) -> bool:
        """
        Early stop the policy
        """
        return self.current_temp <= self.min_temp

    def get_merged_graph(self, graph: Graph, cur_epoch: int) -> Graph:
        """
        Get a merged graph from the current graph
        """
        merged_graph = copy.deepcopy(graph)
        for _ in range(self.n_merge_per_epoch):
            merged_graph.set_req_grad(self.fine_tune_level)
            if merged_graph.relaxed_mergeable_nodes:
                merge_config = merged_graph.relaxed_random_connect()
            else:
                merge_config = merged_graph.random_connect()
            if merge_config and cur_epoch>=0:
                self.history.append(merge_config)
            merged_graph.build_mergeable_nodes()
        return merged_graph

    def get_merge_history(self) -> List[MergeConfig]:
        """
        Get the merging history
        """
        return self.history
    
    def record_history(self, path: str):
        self.export_loc_filename = path
        self.export_history_record = True

    def export_merge_history(self, path="") -> None:
        """
        Export the merging history to a json file
        """
        if path:
            save_file = path
        else:
            save_file = self.export_loc_filename
        with open(save_file, 'w') as f:
            json.dump([asdict(h) for h in self.history], f)
    
    def get_baselines(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the baselines for the accuracy and latency
        """
        cmp_graph = ComputeGraph(
            self.base_graph,
            self.models,
            load_weight=True,
            use_transformer=self.use_transformer,
            device=self.device,
        )
        accuracy_baseline = self.f_accuracy(cmp_graph)
        latency_baseline = self.f_latency(cmp_graph)
        return accuracy_baseline, latency_baseline

    def get_probability(self, acc_delta: torch.Tensor) -> torch.Tensor:
        """
        Get the probability of choosing a graph
        """
        prob = torch.exp(-(1.0-acc_delta) / self.current_temp * self.initial_temp)
        return (1.0 - prob) * sqrt(len(self.elite_candidates) / self.n_elites)

    def get_temperature(self, cur_epoch: int) -> float:
        """
        Get the current temperature
        """
        return self.cooling_mapping[self.cooling_schedule](cur_epoch)
    
    def __linear_cooling(self, cur_epoch: int) -> float:
        """
        Linear cooling schedule
        """
        return self.initial_temp - self.alpha * cur_epoch
    
    def __exponential_cooling(self, cur_epoch: int) -> float:
        """
        Exponential cooling schedule
        """
        return self.initial_temp * self.alpha ** cur_epoch
    
    def __fast_cooling(self, cur_epoch: int) -> float:
        """
        Fast cooling schedule
        """
        return self.initial_temp / cur_epoch
    
    def __boltzmann_cooling(self, cur_epoch: int) -> float:
        """
        Boltzmann cooling schedule
        """
        return self.initial_temp / np.log(cur_epoch)

    def __boltzExp_cooling(self, cur_epoch: int) -> float:
        """
        Boltzmann Exponential cooling schedule
        """
        if cur_epoch <= self.cooling_threshold:
            return self.__boltzmann_cooling(cur_epoch)
        return self.initial_temp * self.alpha ** (cur_epoch - self.initial_temp//2)

    def __constExp_cooling(self, cur_epoch: int) -> float:
        """
        Constant Exponential 1 cooling schedule
        """
        if cur_epoch <= (self.initial_temp // 2):
            return self.initial_temp
        return self.initial_temp * self.alpha ** (cur_epoch - self.initial_temp/2)


class LCBasedSimulatedAnnealingPolicy(SimulatedAnnealingPolicy):
    def __init__(
        self,
        base_graph: Graph,
        models: List[torch.nn.Module],
        f_finetune: Callable,
        f_latency: Callable,
        f_accuracy: Callable,
        load_weight: bool,
        n_merge_per_epoch=2,
        accuracy_tolerence=0.02,
        n_elites=10,
        initial_temp=90,
        min_temp=1,
        alpha=0.9,
        enable_filtering_rules=True,
        cooling_schedule='exponential',
        cooling_threshold=20,
        fine_tune_level=FinetuneLevel.SUBGRAPH,
        use_transformer=False,
        device='cuda'
    ):
        super().__init__(base_graph, models, f_finetune, f_latency, f_accuracy, load_weight, 
                        n_merge_per_epoch, accuracy_tolerence, n_elites, initial_temp, min_temp, alpha, 
                        cooling_schedule, cooling_threshold, fine_tune_level, use_transformer, device)
        # whether or not to enable the rule-based filtering method
        self.enable_filtering_rules = enable_filtering_rules
        # store the capacity of invalid graph
        # rule: if a graph is invalid, more aggressive graph will also be invalid
        # more aggressive: more shared capacity, lower {task specific capacity, total capacity, total capacity for each task}
        self.invalid_capacity_topo = []
        # before the idx, all the capacity features are individual: the lower, the more aggressive
        # after the idx, all the capacity features are shared: the higher, the more aggressive
        self.cap_indiv_share_split_idx = 1 + len(self.models) + len(self.models) # total cap + total task cap + task specific cap
        logging.info(f"Enable the filtering rules to filter graphs: {self.enable_filtering_rules}")

    def filter_by_rules(self, capacity_feat:np.ndarray):
        if_filter = False
        # check if the graph is more aggressive than any invalid graphs
        for cap in self.invalid_capacity_topo:
            if (capacity_feat[:self.cap_indiv_share_split_idx] <= cap[:self.cap_indiv_share_split_idx]).all() \
                    and (capacity_feat[self.cap_indiv_share_split_idx:] >= cap[self.cap_indiv_share_split_idx:]).all() \
                    and (capacity_feat != cap).any():
                if_filter = True
                break
        return if_filter

    def step(self, cur_epoch: int) -> Optional[Tuple[Graph, torch.Tensor]]:
        """
        Given a new mutated graph, first validate it using rule:
            -> compare the capacity with previously invalid graphs -> more aggressive graph will be filtered directly
            -> Use a fine tune method equiped with learning curve extrapolation to predict the final acc drop, and 
                early terminate the graph whose predicted final acc drop cannot meet the threhold
        """
        print("The current number of candidates is {}, the value of P is {}".format(len(self.elite_candidates), self.p))
        logging.info(f'Current Temp: {self.current_temp}, Current probability: {self.p}')
        logging.info(f'Current number of candidates is {len(self.elite_candidates)}')
        logging.info(f'Current number of generated Graphs: {len(self.exist_graphs)}, Topologies: {len(self.exist_topologies)}')

        if not self.elite_candidates or self.p < random.uniform(0, 1):
            print("Optimizing on the Original Graph ...")
            logging.info("Optimizing on the Original Graph ...")
            cur_graph = self.get_merged_graph(self.base_graph, cur_epoch)
            loaded_models = self.models
            for i in range(self.n_merge_per_epoch):
                self.history[cur_epoch*self.n_merge_per_epoch+i].graph_type = 'original_graph'
        else:
            print("Optimizing on Elite Candidate ...") 
            logging.info("Optimizing on Elite Candidate ...")
            idx = random.randint(0, len(self.elite_candidates)-1)
            logging.info(f'Optimizing on elite_candidate {idx}')
            cur_graph = self.get_merged_graph(self.elite_candidates[idx][0], cur_epoch)
            loaded_models = self.exist_graphs[hash(self.elite_candidates[idx][0])]
            for i in range(self.n_merge_per_epoch):
                self.history[cur_epoch*self.n_merge_per_epoch+i].graph_type = 'elite_candidate'
                self.history[cur_epoch*self.n_merge_per_epoch+i].graph_idx = idx

        if self.export_history_record:
            self.export_merge_history()

        cur_graph_hash = hash(cur_graph)
        candidate, cmp_graph = None, None
        if cur_graph_hash not in self.exist_graphs.keys():
            self.exist_graphs[cur_graph_hash] = None
            
            # if current graph topo already exist, check whether it is valid (pass)
            cur_graph_topo_hash = cur_graph.topology_hash()
            if cur_graph_topo_hash in self.exist_topologies and self.exist_topologies[cur_graph_topo_hash] == 1:
                print('Graph Topology is already valid! Skip this iteration ...')
                logging.info('Graph Topology is already valid! Skip this iteration ...')
                return candidate, cmp_graph
            # if not exist, add it into self.exist_topologies
            elif cur_graph_topo_hash not in self.exist_topologies:
                self.exist_topologies[cur_graph_topo_hash] = 0
            
            logging.info(f'Current abs-Graph:\n {cur_graph}')
            total_capacity, branch_capacities = cur_graph.capacity()
            logging.info(f'total capacity: {total_capacity}')
            logging.info('task specific capacity: ')
            for i, cap in enumerate(branch_capacities):
                logging.info(cap)

            cur_cap_feat = cur_graph.export_capacity_numpy()
            # filter by rule
            if self.enable_filtering_rules and self.filter_by_rules(cur_cap_feat):
                print('Current abs-graph is filtered according to the filtering rules')
                logging.critical('Current abs-graph is filtered according to the filtering rules')
                return candidate, cmp_graph

            # Generate a new candidate
            cmp_graph = ComputeGraph(cur_graph, loaded_models, load_weight=self.load_weight, use_transformer=self.use_transformer, device=self.device)
            
            logging.info('Finetune start ...')
            self.f_finetune(cmp_graph, self.accuracy_baseline, self.accuracy_tolerence)
            # Compute accuracy drop
            with HiddenPrints():
                cur_accuracy = self.f_accuracy(cmp_graph)
            # get the max accuracy drop between tasks
            task_acc_delta = self.accuracy_baseline - cur_accuracy
            acc_delta = torch.max(task_acc_delta)
            print('Finetune stop, the accuracy drop is: ', acc_delta.item())

            # Update next temperature
            self.current_temp = self.get_temperature(cur_epoch+1)
            # Update next probability
            self.p = self.get_probability(acc_delta)
            # Compute latency
            cur_latency = self.f_latency(cmp_graph)

            # Log accuracy and latency 
            acc_str = ''
            for i, acc in enumerate(cur_accuracy):
                acc_str += f"net{i+1}: {acc.item()*100}%   "
            logging.info(f'Current Acc: {acc_str}')
            logging.info(f'Finetune ends, the accuracy drop : {acc_delta.item()}, latency : {cur_latency}')

            # Update the candidate
            if acc_delta <= self.accuracy_tolerence:
                logging.info("Acc drop MEET the threshold, add graph to elite candidates ... ")
                self.exist_graphs[cur_graph_hash] = cmp_graph.export_parameters()
                # update current graph topo in self.exist_topologies
                self.exist_topologies[cur_graph_topo_hash] = 1

                if len(self.elite_candidates) < self.n_elites:
                    self.elite_candidates.append((cur_graph, cur_latency))
                    candidate = self.elite_candidates[-1]
                else:
                    # if the elite candidate set is full
                    logging.info("Elite candidate set is full, remove the oldest graph ...")
                    candidate = self.elite_candidates.pop(0) # remove the oldest graph
                    self.elite_candidates.append((cur_graph, cur_latency))
                    candidate = self.elite_candidates[-1]
            else: # update invalid cap topo
                logging.info("Acc drop FAIL to meet the threhold, update invalid capacity list ...")
                self.invalid_capacity_topo.append(cur_cap_feat)
        else:
            print('Graph exists! Skip this epoch ...')
            logging.info('Graph exists! Skip this iteration ...')
        return candidate, cmp_graph 


class RandomSamplingPolicy(SimulatedAnnealingPolicy):
    def step(self, cur_epoch: int) -> Optional[Tuple[Graph, torch.Tensor]]:
        """
        Make a step in the policy given the current state and actions 
            using random sampling
        """
        print("The current number of candidates is {}, the value of P is {}".format(len(self.elite_candidates), self.p))
        logging.info(f'Current Temp: {self.current_temp}, Current probability: {self.p}')
        logging.info(f'Current number of candidates is {len(self.elite_candidates)}')
        logging.info(f'Current number of generated Graphs: {len(self.exist_graphs)}, Topologies: {len(self.exist_topologies)}')

        print("Optimizing on the Original Graph ...")
        logging.info("Optimizing on the Original Graph ...")
        cur_graph = self.get_merged_graph(self.base_graph, cur_epoch)
        loaded_models = self.models
        for i in range(self.n_merge_per_epoch):
            self.history[cur_epoch*self.n_merge_per_epoch+i].graph_type = 'original_graph'

        if self.export_history_record:
            self.export_merge_history()

        cur_graph_hash = hash(cur_graph)
        candidate, cmp_graph = None, None
        if cur_graph_hash not in self.exist_graphs.keys():
            self.exist_graphs[cur_graph_hash] = None
            
            # if current graph topo already exist, check whether it is valid (pass)
            cur_graph_topo_hash = cur_graph.topology_hash()
            if cur_graph_topo_hash in self.exist_topologies and self.exist_topologies[cur_graph_topo_hash] == 1:
                print('Graph Topology is already valid! Skip this iteration ...')
                logging.info('Graph Topology is already valid! Skip this iteration ...')
                return candidate, cmp_graph
            # if not exist, add it into self.exist_topologies
            elif cur_graph_topo_hash not in self.exist_topologies:
                self.exist_topologies[cur_graph_topo_hash] = 0
            
            logging.info(f'Current abs-Graph:\n {cur_graph}')
            total_capacity, branch_capacities = cur_graph.capacity()
            logging.info(f'total capacity: {total_capacity}')
            logging.info('task specific capacity: ')
            for i, cap in enumerate(branch_capacities):
                logging.info(cap)

            # Generate a new candidate
            cmp_graph = ComputeGraph(cur_graph, loaded_models, load_weight=self.load_weight, use_transformer=self.use_transformer, device=self.device)
            
            # logging.info(f'Current cmp-Graph:\n {cmp_graph.nnSeq}')
            # logging.info(f'forward sequence: {cmp_graph.nnSeq_in}')
            
            logging.info('Finetune start ...')
            self.f_finetune(cmp_graph, self.accuracy_baseline, self.accuracy_tolerence)
            # Compute accuracy drop
            with HiddenPrints():
                cur_accuracy = self.f_accuracy(cmp_graph)
            # get the avg accuracy drop between tasks
            # acc_delta = sum(self.accuracy_baseline - cur_accuracy) / len(self.accuracy_baseline)
            # get the max accuracy drop between tasks
            task_acc_delta = self.accuracy_baseline - cur_accuracy
            acc_delta = torch.max(task_acc_delta)
            print('Finetune stop, the accuracy drop is: ', acc_delta.item())

            # Analyze
            self.analyze_accuracy_drop.append(task_acc_delta.tolist())
            self.analyze_task_total_capacity.append([b.total for b in branch_capacities])
            self.analyze_task_specific_capacity.append([b.individual for b in branch_capacities])

            # Compute latency
            cur_latency = self.f_latency(cmp_graph)

            # Log accuracy and latency 
            acc_str = ''
            for i, acc in enumerate(cur_accuracy):
                acc_str += f"net{i+1}: {acc.item()*100}%   "
            logging.info(f'Current Acc: {acc_str}')
            logging.info(f'Finetune ends, the accuracy drop : {acc_delta.item()}, latency : {cur_latency}')

            # Update the candidate
            if acc_delta <= self.accuracy_tolerence:
                logging.info("Acc drop MEET the threshold, add graph to elite candidates ... ")
                self.exist_graphs[cur_graph_hash] = cmp_graph.export_parameters()
                # update current graph topo in self.exist_topologies
                self.exist_topologies[cur_graph_topo_hash] = 1

                if len(self.elite_candidates) < self.n_elites:
                    self.elite_candidates.append((cur_graph, cur_latency))
                    candidate = self.elite_candidates[-1]
                else:
                    # if the elite candidate set is full
                    logging.info("Elite candidate set is full, remove the oldest graph ...")
                    candidate = self.elite_candidates.pop(0) # remove the oldest graph
                    self.elite_candidates.append((cur_graph, cur_latency))
                    candidate = self.elite_candidates[-1]
        else:
            print('Graph exists! Skip this epoch ...')
            logging.info('Graph exists! Skip this iteration ...')
        return candidate, cmp_graph 


class FilterBasedSimulatedAnnealingPolicy(SimulatedAnnealingPolicy):
    def __init__(
        self,
        base_graph: Graph,
        models: List[torch.nn.Module],
        f_finetune: Callable,
        f_latency: Callable,
        f_accuracy: Callable,
        load_weight: bool,
        n_merge_per_epoch=2,
        accuracy_tolerence=0.02,
        n_elites=10,
        initial_temp=90,
        min_temp=1,
        alpha=0.9,
        filtering_model_type="xgbc",
        num_starting_topo=32,
        cooling_schedule='exponential',
        cooling_threshold=20,
        fine_tune_level=FinetuneLevel.SUBGRAPH,
        use_transformer=False,
        device='cuda'
    ):
        super().__init__(base_graph, models, f_finetune, f_latency, f_accuracy, load_weight, 
                        n_merge_per_epoch, accuracy_tolerence, n_elites, initial_temp, min_temp, alpha, 
                        cooling_schedule, cooling_threshold, fine_tune_level, use_transformer, device)
        self.num_starting_topo = num_starting_topo
        # init the filtering model
        self.filtering_model_type = filtering_model_type
        self.filtering_model = self.init_filtering_model()
        self.filtering_model_params = {}
        # generate starting topo given self.num_starting_topo, store in the topo_hash_list
        self.exist_graphs: Dict[GraphHash, Dict[NodeIndex, nn.Module]] = {}
        self.exist_topologies: DefaultDict[TopoHash, bool] = defaultdict(dict)
        # training data X is the capacity of the graph
        self.train_graph: List[Graph] = []
        self.train_X = self.sample_starting_topo()
        # training label y is whether the graph is valid given acc drop (pass:1, fail:0)
        self.train_y = [0 for _ in self.train_X]

        logging.info(f"Filtering model init: {self.filtering_model_type}, Num of starting topo: {self.num_starting_topo}")

    def init_filtering_model(self):
        # initialize the filtering model given the args
        if self.filtering_model_type == "gbc":
            return GradientBoostingClassifier()
        elif self.filtering_model_type == 'xgbc':
            return XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
        elif self.filtering_model_type == 'rfc':
            return RandomForestClassifier()

    def fit_filtering_model(self) -> None:
        fit_X, fit_y = np.array(self.train_X), np.array(self.train_y)
        logging.critical("Update the filtering model ...")
        logging.info(f'Current size of training set: {len(fit_X)}')
        # train the model using selected HP
        self.filtering_model = self.init_filtering_model()
        self.filtering_model.set_params(**self.filtering_model_params)
        self.filtering_model.fit(fit_X, fit_y)
        logging.critical("Filtering model is updated")

    def filtering_model_params_search(self) -> None:
        fit_X, fit_y = np.array(self.train_X), np.array(self.train_y)
        logging.critical("Searching the best Hyper Params for the filtering model ...")
        logging.info(f'Current size of training set: {len(fit_X)}')
        # cross validation (leave one out) to select HP
        if self.filtering_model_type in ["gbc", "xgbc"]:
            hyper_params = {'n_estimators': stats.randint(5,50),
                            'learning_rate': stats.uniform(0.01, 10),
                            'max_depth': [3,4,5,6,7,8,9,10]}
            n_iter = 200
        elif self.filtering_model_type in ["rfc"]:
            hyper_params = {'n_estimators': stats.randint(5,50),
                            'max_depth': [3,4,5,6,7,8,9,10]}
            n_iter = 100
        self.filtering_model = self.init_filtering_model()
        cv = RandomizedSearchCV(self.filtering_model, hyper_params, cv=len(self.train_X), 
                                n_iter=n_iter, scoring='f1')
        cv.fit(fit_X, fit_y)
        logging.critical(f'Hyper params selected: {cv.best_params_}')
        self.filtering_model_params = cv.best_params_

    def sample_starting_topo(self) -> List[np.ndarray]:
        # TODO: use random sampling
        train_X = []
        # # estimate the total number of graph that are generated from the original graph,
        # # OR estimate the # samples we need to ensure the variance of the distribution of the searching space
        diff = 10
        est_num = self.num_starting_topo * diff
        # do random connect on the original graph, then sort given the total capacity
        tmp_capacity, tmp_train_graph = [], []
        tmp_exist_graph, tmp_exist_topo = [], []
        for e in range(est_num):
            logging.info(f'Sampling the {e}th graph ...')
            new_graph = self.get_merged_graph(self.base_graph, -1)
            new_graph_hash = hash(new_graph)
            new_graph_topo_hash = new_graph.topology_hash()
            if new_graph_topo_hash not in tmp_exist_topo:
                # get the capacity 
                tmp_capacity.append(new_graph.export_capacity_numpy())
                tmp_train_graph.append(new_graph)
                tmp_exist_graph.append(new_graph_hash)
                tmp_exist_topo.append(new_graph_topo_hash)
        sorted_idx = np.argsort([v[0] for v in tmp_capacity])
        # tmp_capacity.sort(key=lambda x:x[0])
        # select self.num_starting_topo samples uniformly
        if len(tmp_capacity) <= self.num_starting_topo:
            train_X = tmp_capacity
            self.train_graph = tmp_train_graph
            for i in len(self.train_graph):
                self.exist_graphs[tmp_exist_graph[i]] = None
                self.exist_topologies[tmp_exist_topo[i]] = 0
        else:
            select_space = np.linspace(0,len(tmp_capacity)-1, self.num_starting_topo, dtype=int)
            for i in select_space:
                train_X.append(tmp_capacity[sorted_idx[i]])
                self.train_graph.append(tmp_train_graph[sorted_idx[i]])
                self.exist_graphs[tmp_exist_graph[sorted_idx[i]]] = None
                self.exist_topologies[tmp_exist_topo[sorted_idx[i]]] = 0
        logging.critical(f'Try to sample {self.num_starting_topo} topo, successfully sampled {len(train_X)} topo')
        self.num_starting_topo = len(train_X)
        return train_X

    def update_training_data(self, new_X: np.ndarray, new_y: int) -> Tuple[int, int]:
        # TODO: use TopoHash to check if the new graph is already in the training set
        num_data_updated, num_data_added = 0, 0
        topo_exist = False
        # after train a new graph, update the training set
        for idx, tr in enumerate(self.train_X):
            if (new_X == tr).all():
                # if it is an existing topo, update the label (pass or fail)
                if new_y == 1 and self.train_y[idx] == 0: 
                    # if this new graph pass and existing topo fails, then update
                    self.train_y[idx] = new_y
                    num_data_updated += 1
                topo_exist = True
                break
        # if it is a new topo, add to the training set
        if not topo_exist:
            self.train_X.append(new_X)
            self.train_y.append(new_y)
            num_data_added += 1
        logging.critical(f'{num_data_updated} topo updated, {num_data_added} topo added, current size of training set: {len(self.train_X)}')
        return num_data_updated, num_data_added

    def step(self, cur_epoch: int) -> Optional[Tuple[Graph, torch.Tensor]]:
        """
        stage 0: when cur_epoch < self.num_starting_topo, train graph from self.train_graph
        when all the graph in self.train_graph have been trained, fit the filtering model
        stage 1: adopt simulated annealing: Generate new graph from original graph or elite candidate -> check whether graph exists
                                    -> if graph exist, skip the iteration
                                    -> if graph not exist, check whether the topo exists in self.exist_topologies
                                    -> if topo not exist, filtering model determines ...
                                    -> if topo exist, check whether topo is passed or failed -> if failed, filtering model 
                                    determines the graph is worth training or not -> if true, train the graph, update the self.train_X
                                    and self.train_y -> update the filtering model
        tips: different topo: the topo structures are different
              different graph with different topo: both the topo and init_weight should be different
              different graph with same topo: the topo structures are the same, but have different init_weight
        """
        candidate, cmp_graph = None, None

        if cur_epoch < self.num_starting_topo:
            stage = 0
            cur_graph = self.train_graph.pop(0)
            new_X = self.train_X[cur_epoch]
            cur_graph_hash = hash(cur_graph)
            cur_graph_topo_hash = cur_graph.topology_hash()
            loaded_models = self.models
            print(f"Select and Train graph from starting topo, topo remaining {len(self.train_graph)}")
            logging.info(f"Select and Train graph from starting topo, starting topo remaining: {len(self.train_graph)}")

            logging.info(f'Current abs-Graph:\n {cur_graph}')
            total_capacity, branch_capacities = cur_graph.capacity()
            logging.info(f'total capacity: {total_capacity}')
            logging.info('task specific capacity: ')
            for i, cap in enumerate(branch_capacities):
                logging.info(cap)
        
        else:
            stage = 1
            print("The current number of candidates is {}, the value of P is {}".format(len(self.elite_candidates), self.p))
            logging.info(f'Current Temp: {self.current_temp}, Current probability: {self.p}')
            logging.info(f'Current number of candidates is {len(self.elite_candidates)}')
            if not self.elite_candidates or self.p < random.uniform(0, 1):
                print("Optimizing on the Original Graph ...")
                logging.info("Optimizing on the Original Graph ...")
                cur_graph = self.get_merged_graph(self.base_graph, cur_epoch)
                loaded_models = self.models
            else:
                print("Optimizing on Elite Candidate ...") 
                logging.info("Optimizing on Elite Candidate ...")
                idx = random.randint(0, len(self.elite_candidates)-1)
                logging.info(f'Optimizing on elite_candidate {idx}')
                cur_graph = self.get_merged_graph(self.elite_candidates[idx][0], cur_epoch)
                loaded_models = self.exist_graphs[hash(self.elite_candidates[idx][0])]
            new_X = cur_graph.export_capacity_numpy()
            cur_graph_hash = hash(cur_graph)
            cur_graph_topo_hash = cur_graph.topology_hash()
            # print(new_X)
            
            # TODO: Refactor unnecessary else clauses
            # check whether the graph exists already
            if cur_graph_hash in self.exist_graphs:
                print('Graph exists! Skip this iteration ...')
                logging.info('Graph exists! Skip this iteration ...')
                return candidate, cmp_graph
            if cur_graph_topo_hash in self.exist_topologies: # topo exists
                if self.exist_topologies[cur_graph_topo_hash] == 1: # topo passes
                    print('Topo exists and valid! skip this iteration')
                    logging.info('Topo exists and valid! skip this iteration')
                    self.exist_graphs[cur_graph_hash] = None
                    return candidate, cmp_graph
            else: # topo not exists
                self.exist_topologies[cur_graph_topo_hash] = 0

            logging.info(f'Current abs-Graph:\n {cur_graph}')
            total_capacity, branch_capacities = cur_graph.capacity()
            logging.info(f'total capacity: {total_capacity}')
            logging.info('task specific capacity: ')
            for i, cap in enumerate(branch_capacities):
                logging.info(cap)
            
            # topo not exists, or topo exists but failed
            print('A new graph is generated, determining by filtering model ...')
            logging.info('A new graph is generated, determining by filtering model ...')
            pred_valid = self.filtering_model.predict(new_X.reshape(1,-1))
            if pred_valid == 0: # predict to be failed
                print('Predict to be failed, skip this iteration ...')
                logging.info('Predict to be failed, skip this iteration ...')
                return candidate, cmp_graph
            else: # predict to be passed
                print('Predict to be passed, process the graph ...')
                logging.info('Predict to be passed, process the graph ...')
                self.exist_graphs[cur_graph_hash] = None
            
        cmp_graph = ComputeGraph(cur_graph, loaded_models, load_weight=self.load_weight, use_transformer=self.use_transformer, device=self.device)
        logging.info('Finetune start ...')
        self.f_finetune(cmp_graph, self.accuracy_baseline, self.accuracy_tolerence)
        # Compute the accuracy drop
        with HiddenPrints():
            cur_accuracy = self.f_accuracy(cmp_graph)
        task_acc_delta = self.accuracy_baseline - cur_accuracy
        acc_delta = torch.max(task_acc_delta)
        print('Finetune stop, the accuracy drop is: ', acc_delta.item())

        # compute latency
        cur_latency = self.f_latency(cmp_graph)
        # Log accuracy and latency
        acc_str = ''
        for i, acc in enumerate(cur_accuracy):
            acc_str += f"net{i+1}: {acc.item()*100}%   "
        logging.info(f'Current Acc: {acc_str}')
        logging.info(f'Finetune ends, the accuracy drop : {acc_delta.item()}, latency : {cur_latency}')

        # Update the candidate, self.train_X, self.train_y
        new_y = 0
        if acc_delta <= self.accuracy_tolerence:
            new_y = 1 # pass
            logging.info("Acc drop MEET the threshold, add graph to elite candidates ... ")
            self.exist_graphs[cur_graph_hash] = cmp_graph.export_parameters()
            self.exist_topologies[cur_graph_topo_hash] = 1

            if len(self.elite_candidates) < self.n_elites:
                self.elite_candidates.append((cur_graph, cur_latency))
                candidate = self.elite_candidates[-1]
            else:
                # if the elite candidate set is full
                logging.info("Elite candidate set is full, remove the oldest graph ...")
                candidate = self.elite_candidates.pop(0) # remove the oldest graph
                self.elite_candidates.append((cur_graph, cur_latency))
                candidate = self.elite_candidates[-1]
        
        num_data_updated, num_data_added = self.update_training_data(new_X, new_y)

        if stage == 0 and not self.train_graph:
            self.filtering_model_params_search()
            self.fit_filtering_model()

        if stage == 1 and (num_data_updated + num_data_added)==0:
            self.fit_filtering_model()

        return candidate, cmp_graph


class ManualSimulatedAnnealingPolicy(SimulatedAnnealingPolicy):

    def load_history_from_obj(self, history: List[MergeConfig]) -> None:
        self.history = history

    def load_history_from_json(self, path: str) -> None:
        with open(path) as f:
            load_history = json.load(f)
            for load_h in load_history:
                self.history.append(MergeConfig(
                    input_sig=torch.Size(load_h['input_sig']),
                    node1_idx=tuple(load_h['node1_idx']),
                    node2_idx=tuple(load_h['node2_idx']),
                    direction=Direction(load_h['direction']),
                    relation=Relation(load_h['relation']),
                    graph_type=str(load_h['graph_type']),
                    graph_idx=int(load_h['graph_idx'])
                ))

    def get_merged_graph(self, graph: Graph, cur_epoch: int) -> Graph:
        """
        overload get_merged_graph function
        read merged nodes and info from history and do force connect
        (compared to random connect)
        """
        epoch = cur_epoch * self.n_merge_per_epoch
        merged_graph = copy.deepcopy(graph)
        for i in range(self.n_merge_per_epoch):
            merged_graph.set_req_grad(self.fine_tune_level)
            merge_config = self.history[epoch+i]
            if merge_config.input_sig[0] != -1:
                merged_graph.force_connect(merge_config)
            merged_graph.build_mergeable_nodes()
        return merged_graph

    def step(self, cur_epoch: int) -> Optional[Tuple[Graph, torch.Tensor]]:
        """
        Overload the step function
        read merged_graph from history, and to optimization on the merged_graph
        (compared to doing optimization with probability p)
        """
        print("The current number of candidates is {}, the value of P is {}".format(len(self.elite_candidates), self.p))
        logging.info(f'Current Temp: {self.current_temp}, Current probability: {self.p}')
        logging.info(f'Current number of candidates is {len(self.elite_candidates)}')

        merge_config = self.history[cur_epoch * self.n_merge_per_epoch]
        if merge_config.graph_type == 'original_graph':
            print("Optimizing on the Original Graph ...")
            logging.info("Optimizing on the Original Graph ...")
            cur_graph = self.get_merged_graph(self.base_graph, cur_epoch)
            loaded_models = self.models
        elif merge_config.graph_type == 'elite_candidate':
            print("Optimizing on Elite Candidate ...") 
            logging.info("Optimizing on Elite Candidate ...")
            idx = merge_config.graph_idx
            logging.info(f'Optimizing on elite_candidate {idx}')
            cur_graph = self.get_merged_graph(self.elite_candidates[idx][0], cur_epoch)
            loaded_models = self.exist_graphs[hash(self.elite_candidates[idx][0])]
        else:
            raise ValueError("Graph Type must be either 'original_graph' or 'elite_candidate'")

        cur_graph_hash = hash(cur_graph)
        candidate, cmp_graph = None, None
        if cur_graph_hash not in self.exist_graphs.keys():
            self.exist_graphs[cur_graph_hash] = None
            
            cur_graph_topo_hash = cur_graph.topology_hash()
            if cur_graph_topo_hash in self.exist_topologies and self.exist_topologies[cur_graph_topo_hash] == 1:
                print('Graph Topology is already valid! Skip this iteration ...')
                logging.info('Graph Topology is already valid! Skip this iteration ...')
                return None, None
            # if not exist, add it into self.exist_topologies
            elif cur_graph_topo_hash not in self.exist_topologies:
                self.exist_topologies[cur_graph_topo_hash] = 0
            
            logging.info(f'Current abs-Graph:\n {cur_graph}')

            # Generate a new candidate
            cmp_graph = ComputeGraph(cur_graph, loaded_models, load_weight=self.load_weight, use_transformer=self.use_transformer, device=self.device)
            logging.info('Finetune start ...')
            self.f_finetune(cmp_graph, self.accuracy_baseline, self.accuracy_tolerence)
            # Compute accuracy drop
            with HiddenPrints():
                cur_accuracy = self.f_accuracy(cmp_graph)
            # get the avg accuracy drop between tasks
            # acc_delta = sum(self.accuracy_baseline - cur_accuracy) / len(self.accuracy_baseline)
            # get the max accuracy drop between tasks
            acc_delta = torch.max(self.accuracy_baseline - cur_accuracy)
            print('Finetune stop, the accuracy drop is: ', acc_delta.item())

            # Update next temperature
            self.current_temp = self.get_temperature(cur_epoch+1)
            # Update next probability
            self.p = self.get_probability(acc_delta)
            # Compute latency
            cur_latency = self.f_latency(cmp_graph)

            # Log accuracy and latency 
            str = ''
            for i, acc in enumerate(cur_accuracy):
                str += f"net{i+1}: {acc.item()*100}%   "
            logging.info(f'Current Acc: {str}')
            logging.info(f'Finetune ends, the accuracy drop : {acc_delta.item()}, latency : {cur_latency}')

            # Update the candidate
            if acc_delta <= self.accuracy_tolerence:
                logging.info("Acc drop MEET the threshold, add graph to elite candidates ... ")
                self.exist_graphs[cur_graph_hash] = cmp_graph.export_parameters()
                # update current graph topo in self.exist_topologies
                self.exist_topologies[cur_graph_topo_hash] = 1

                if len(self.elite_candidates) < self.n_elites:
                    self.elite_candidates.append((cur_graph, cur_latency))
                    candidate = self.elite_candidates[-1]
                else:
                    # if the elite candidate set is full
                    logging.info("Elite candidate set is full, remove the oldest graph ...")
                    candidate = self.elite_candidates.pop(0) # remove the oldest graph
                    self.elite_candidates.append((cur_graph, cur_latency))
                    candidate = self.elite_candidates[-1]
        else:
            print('Graph exists! Skip this epoch ...')
            logging.info('Graph exists! Skip this iteration ...')
        return candidate, cmp_graph 


class IterativePolicy(Policy):

    def __init__(
        self,
        base_graph: Graph,
        models: List[torch.nn.Module],
        f_finetune: Callable,
        f_latency: Callable,
        f_accuracy: Callable,
        load_weight: bool,
        accuracy_tolerence=0.02,
        fine_tune_level=FinetuneLevel.SUBGRAPH,
        device='cuda'
    ):
        super(IterativePolicy, self).__init__()

        # Optimization schedule
        self.base_graph = base_graph
        self.models = models
        self.accuracy_tolerence = accuracy_tolerence
        self.elite_candidates = []
        self.exist_graphs: Dict[GraphHash, Graph] = {}
        self.exist_topologies: DefaultDict[TopoHash, Dict[GraphHash, Tuple[float, ...]]] = defaultdict(dict)
        self.device = device
        self.load_weight = load_weight

        # Evaluation functions
        self.f_finetune = f_finetune
        self.f_latency = f_latency
        self.f_accuracy = f_accuracy

        # sub-graph finetune = FinetuneLevel.SUBGRAPH, entire-graph finetune = FinetuneLevel.FULLGRAPH
        self.fine_tune_level = fine_tune_level

        # Baselines
        self.accuracy_baseline, self.latency_baseline = self.get_baselines()

        self.candidates = []
        self.next_candidates = []
        self.iter_graph(self.base_graph)
    
    def get_baselines(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the baselines for the accuracy and latency
        """
        # cmp_graph = ComputeGraph(
        #     self.base_graph,
        #     self.models,
        #     load_weight=True,
        #     device=self.device
        # )
        accuracy_baseline = 0
        latency_baseline = 0
        # accuracy_baseline = self.f_accuracy(cmp_graph)
        # latency_baseline = self.f_latency(cmp_graph)
        return accuracy_baseline, latency_baseline

    def step(self, cur_epoch: int) -> Optional[Tuple[Graph, torch.Tensor]]:
        """
        Make a step in the policy iteratively
        """
        if not self.candidates and not self.next_candidates:
            return None, None
        elif not self.candidates:
            self.candidates, self.next_candidates = self.next_candidates, []

        cur_graph: Graph = self.candidates.pop(0)
        cur_graph_hash = hash(cur_graph)

        logging.info(f'Current abs-Graph:\n {cur_graph}')
        total_capacity, branch_capacities = cur_graph.capacity()
        logging.info(f'total capacity: {total_capacity}')
        logging.info('task specific capacity: ')
        for i, cap in enumerate(branch_capacities):
            logging.info(cap)

        # Generate new candidates
        self.iter_graph(cur_graph)
        # loaded_models = self.models
        cmp_graph = None
        # cmp_graph = ComputeGraph(cur_graph, loaded_models, load_weight=self.load_weight, device=self.device)
        # logging.info('Finetune start ...')
        # self.f_finetune(cmp_graph, self.accuracy_baseline, self.accuracy_tolerence)
        # Compute accuracy drop
        # with HiddenPrints():
            # cur_accuracy = self.f_accuracy(cmp_graph)
        cur_accuracy = self.accuracy_baseline
        # get the avg accuracy drop between tasks
        # acc_delta = sum(self.accuracy_baseline - cur_accuracy) / len(self.accuracy_baseline)
        # get the max accuracy drop between tasks
        # acc_delta = torch.max(self.accuracy_baseline - cur_accuracy)
        acc_delta = self.accuracy_baseline - cur_accuracy
        # print('Finetune stop, the accuracy drop is: ', acc_delta.item())

        # Compute latency
        # cur_latency = self.f_latency(cmp_graph)
        cur_latency = self.latency_baseline

        # Log accuracy and latency 
        acc_str = ''
        # for i, acc in enumerate(cur_accuracy):
        #     acc_str += f"net{i+1}: {acc.item()*100}%   "
        # logging.info(f'Current Acc: {acc_str}')
        # logging.info(f'Finetune ends, the accuracy drop : {acc_delta.item()}, latency : {cur_latency}')

        # Update the candidate
        met_acc_threshold = acc_delta <= self.accuracy_tolerence
        self.exist_topologies[cur_graph.topology_hash()][cur_graph_hash] = (met_acc_threshold, cur_accuracy, cur_latency)
        if met_acc_threshold:
            logging.info("Acc drop MEET the threshold, add graph to elite candidates ... ")
            self.exist_graphs[cur_graph_hash] = None

        # Log the topology overview every 10 epochs
        if cur_epoch % 50 == 0:
            logging.critical(f'Total topologies: {len(self.exist_topologies)}')
            logging.critical(f'Total graphs: {len(self.exist_graphs)}')
            for i, graph_dict in enumerate(self.exist_topologies.values()):
                logging.critical(f'Topology {i}: {len(graph_dict)} initializations')
                # for j, graph_info in enumerate(graph_dict.values()):
                #     if graph_info is not None:
                #         graph_acc_status, graph_acc, graph_latency = graph_info
                #         acc_status = 'PASS' if graph_acc_status else 'FAIL'
                #         logging.critical(f'\tInit {j}: Acc Status: {acc_status}, Acc: {graph_acc}, Latency: {graph_latency}')
                #     else:
                #         logging.critical(f'\tInit {j}: Acc Status: N/A, Acc: N/A, Latency: N/A')
        return (cur_graph, cur_latency), cmp_graph

    def early_stop(self) -> bool:
        return False
    
    def iter_graph(self, graph: Graph) -> None:
        graph.build_mergeable_nodes()
        for input_sig, nodes in graph.mergeable_nodes.items():
            if len(nodes) > 1:
                combos = list(combinations(nodes, 2))
                for node1, node2 in combos:
                    # logging.info(f'Merging {node1.op_index} and {node2.op_index}')
                    # logging.info('Try Left Merge')
                    left_merged_graph = self.get_merged_graph(
                        graph, input_sig, node1, node2, Direction.LEFT
                    )
                    # logging.info('Try Right Merge')
                    right_merged_graph = self.get_merged_graph(
                        graph, input_sig, node1, node2, Direction.RIGHT
                    )
                    if left_merged_graph is not None:
                        left_graph_hash = hash(left_merged_graph)
                        if left_graph_hash not in self.exist_graphs:
                            self.next_candidates.append(left_merged_graph)
                            self.exist_graphs[left_graph_hash] = None
                            self.exist_topologies[left_merged_graph.topology_hash()][left_graph_hash] = None
                        else:
                            pass
                            # logging.warning('Left Graph exists!')
                            # logging.warning(f'Current Graph: \n{str(left_merged_graph)}')
                            # logging.warning(f'Existing Graph: \n{str(self.exist_graphs[left_graph_hash])}')
                    else:
                        pass
                        # logging.warning('Left Graph is invalid!')
                    if right_merged_graph is not None:
                        right_graph_hash = hash(right_merged_graph)
                        if right_graph_hash not in self.exist_graphs:
                            self.next_candidates.append(right_merged_graph)
                            self.exist_graphs[right_graph_hash] = None
                            self.exist_topologies[right_merged_graph.topology_hash()][right_graph_hash] = None
                        else:
                            pass
                            # logging.warning('Right Graph exists!')
                            # logging.warning(f'Current Graph: \n{str(right_merged_graph)}')
                            # logging.warning(f'Existing Graph: \n{str(self.exist_graphs[right_graph_hash])}')
                    else:
                        pass
                        # logging.warning('Right Graph is invalid!')

    def get_merged_graph(
            self, graph: Graph, input_sig: InputSig,
            node1: Node, node2: Node,
            direction: Direction
        ) -> Optional[Graph]:
        merged_graph = copy.deepcopy(graph)
        merge_config = MergeConfig(
            input_sig=input_sig,
            node1_idx=node1.op_index,
            node2_idx=node2.op_index,
            direction=direction,
            relation=merged_graph.check_parent_child_violation(node1, node2)
        )
        merge_config = merged_graph.manual_connect(merge_config)
        if merge_config.input_sig[0] != -1:
            merged_graph.set_req_grad(self.fine_tune_level)
            merged_graph.build_mergeable_nodes()
            return merged_graph
        else:
            return None
