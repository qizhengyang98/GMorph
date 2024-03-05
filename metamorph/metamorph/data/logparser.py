from typing import Tuple, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import itertools
from ast import literal_eval


def task_share_combinations(num_task):
    task_list = [i for i in range(1,num_task+1)]
    task_share_pattern = []
    for i in range(1,num_task+1):
        task_share_pattern += list(itertools.combinations(task_list, i))
    return task_share_pattern

"""
Used to analyze data in the generated logs
"""
class LogParser:
    DATE_LEN = 10
    TIME_LEN = 8
    LOG_TYPE = ('INFO', 'WARNING', 'ERROR', 'UNKNOWN')

    def __init__(self, path: str) -> None:
        self.log_path = path
        self.log = None
        self.n_models = 0
    
    def base_parser(self, line: str) -> Tuple[Optional[datetime], str, str]:
        if line[0].isdigit():
            date_str, line = line.split(' ', 1)
            assert len(date_str) == self.DATE_LEN
            time_str, line = line.split(' ', 1)
            assert len(time_str) == self.TIME_LEN
            log_time = datetime.strptime(" ".join([date_str, time_str]), r"%Y-%m-%d %H:%M:%S")

            log_type, line = line.split(' ', 1)
            assert log_type in self.LOG_TYPE

            log_msg = line.strip()
        else:
            log_time = None
            log_type = 'UNKNOWN'
            log_msg = line.strip()
        return log_time, log_type, log_msg
    
    def graph_parser(self) -> str:
        graph_str = []
        while line := self.log.readline():
            if line == '\n':
                break
            graph_str.append(line)
        return ''.join(graph_str)
    
    def accuracy_parser(self, line: str) -> List[float]:
        accuracies = line.split(' ', 2)[-1].strip()
        accuracies = accuracies.split('   ')
        accuracies = [acc.split(' ', 1)[-1][:-2] for acc in accuracies]
        accuracies = [float(acc)/100 for acc in accuracies]
        return accuracies

    def latency_parser(self, line: str) -> float:
        latency = line.split(' ', 2)[-1].strip()
        latency = float(latency)
        return latency
    
    def acc_latency_parser(self, line: str) -> float:
        latency = line.rsplit(' ', 1)[-1].strip()
        latency = float(latency)
        return latency
    
    def compile_time_parser(self, line: str) -> float:
        compile_time = line.rsplit(' ', 1)[-1].strip()
        compile_time = float(compile_time)
        return compile_time
    
    def capacity_parser(self, line: str) -> Tuple[int, List[Tuple[int, int, int]]]:
        total_capacity = int(line.rsplit(' ', 1)[-1].strip())
        task_capacities = []
        task_counter = self.n_models
        task_shared_capacities = []

        while line := self.log.readline():
            _, _, log_msg = self.base_parser(line)
            if log_msg.startswith("Task_id"):
                task_total, task_individual, task_shared = 0, 0, 0
            elif log_msg.startswith("Total"):
                task_total_str, log_msg = log_msg.split(' ', 2)[1:]
                task_individual, log_msg = log_msg.split(' ', 2)[1:]
                task_shared = log_msg.split(' ', 1)[-1]

                task_total = int(task_total_str.strip(r'\n ,)'))
                task_individual = int(task_individual.strip(r'\n ,)'))
                task_shared = int(task_shared.strip(r'\n ,)'))
                task_capacities.append((task_total, task_individual, task_shared))
            elif log_msg.startswith("Details"):
                # shared capacity details 
                tmp_task_shared_capacities = {i:0 for i in task_share_combinations(self.n_models)}
                while details := self.log.readline():
                    if details == '\n':
                        break
                    else:
                        task_share_pattern, task_share_cap = details.split(':')
                        tmp_task_shared_capacities[literal_eval(task_share_pattern.strip())] = int(task_share_cap.strip())
                task_shared_capacities.append(tmp_task_shared_capacities)
                task_counter -= 1
            else:
                # All uncaught logs can be ignored
                pass
            
            if task_counter == 0:
                break
        return total_capacity, task_capacities, task_shared_capacities
    
    def merge_parser(self) -> Tuple[str, str]:
        node1_info = self.log.readline().rsplit(")", 1)[-1].strip()
        node2_info = self.log.readline().rsplit(")", 1)[-1].strip()
        return node1_info, node2_info

    def original_model_parser(self) -> Tuple[List[float], float]:
        accuracies = None
        latency = None

        while line := self.log.readline():
            _, _, log_msg = self.base_parser(line)
            if log_msg.startswith("Original Graph"):
                self.graph_parser()
            elif log_msg.startswith("Baselines Acc"):
                accuracies = self.accuracy_parser(log_msg)
            elif log_msg.startswith("Baseline Latency"):
                latency = self.latency_parser(log_msg)
            elif log_msg.startswith("--"):
                # New optimization iteration
                break
            else:
                # All uncaught logs can be ignored
                pass
        return accuracies, latency

    def opt_iteration_parser(self) -> Tuple:
        node1_info, node2_info = None, None
        total_capacity, task_capacities, task_shared_capacities = None, None, None
        accuracies = None
        latency, compile_time = None, None

        while line := self.log.readline():
            _, _, log_msg = self.base_parser(line)
            if log_msg.startswith("Merging"):
                node1_info, node2_info = self.merge_parser()
            elif log_msg.startswith("Current abs-Graph"):
                self.graph_parser()
            elif log_msg.startswith("total capacity"):
                total_capacity, task_capacities, task_shared_capacities = self.capacity_parser(log_msg)
            elif log_msg.startswith("Current Acc:"):
                accuracies = self.accuracy_parser(log_msg)
            elif log_msg.startswith("Finetune ends"):
                latency = self.acc_latency_parser(log_msg)
            elif log_msg.startswith("Compiling time"):
                compile_time = self.compile_time_parser(log_msg)
            elif log_msg.startswith("--"):
                # New optimization iteration
                break
            else:
                # All uncaught logs can be ignored
                pass
        return node1_info, node2_info, total_capacity, task_capacities, task_shared_capacities, accuracies, latency, compile_time

    def parse(self) -> None:
        self.log = open(self.log_path, 'r', encoding='UTF8')
        counter = 1
        base_accuracies, base_latency = self.original_model_parser()
        self.n_models = len(base_accuracies)
        print(f"Base Info: ")
        print(f"    # of models: {self.n_models}")
        print(f"    Accuracies: {base_accuracies}")
        print(f"    Latency: {base_latency}")
        print()

        while line := self.log.readline():
            if line == '\n' or line.strip().endswith("--"):
                break
            else:
                node1_info, node2_info, total_capacity, task_capacities, task_shared_capacities, accuracies, latency, compile_time = self.opt_iteration_parser()
                print(f"Optimization Iteration{counter}: ")
                print(f"    Node1: {node1_info}, Node2: {node2_info}")
                print(f"    Total Capacity: {total_capacity}")
                print(f"    Task Capacities: {task_capacities}")
                print(f"    Task Shared Capaciites: {task_shared_capacities}")
                print(f"    Accuracies: {accuracies}")
                print(f"    Latency: {latency}")
                print(f"    Compile Time: {compile_time}")
                print()
                counter += 1
        self.log.close()
    
    def append_base_info(self, log_dict: dict, accuracies: List[float], latency: float) -> None:
        log_dict['total_capacity'].append(None)
        log_dict['latency'].append(latency)
        log_dict['compile_time'].append(None)
        for i in range(self.n_models):
            log_dict[f'task{i+1}_total'].append(None)
            log_dict[f'task{i+1}_individual'].append(None)
            log_dict[f'task{i+1}_shared'].append(None)
            log_dict[f'task{i+1}_accuracy'].append(accuracies[i])
            for k in task_share_combinations(self.n_models):
                log_dict[f'task{i+1}_shared_{k}'].append(None)
    
    def append_opt_info(
        self, log_dict: dict, 
        node1_info, node2_info, 
        total_capacity, task_capacities, task_shared_capacities,
        accuracies, latency, compile_time
    ) -> None:
        log_dict['total_capacity'].append(total_capacity)
        log_dict['latency'].append(latency)
        log_dict['compile_time'].append(compile_time)
        for i in range(self.n_models):
            if task_capacities:
                log_dict[f'task{i+1}_total'].append(task_capacities[i][0])
                log_dict[f'task{i+1}_individual'].append(task_capacities[i][1])
                log_dict[f'task{i+1}_shared'].append(task_capacities[i][2])
            else:
                log_dict[f'task{i+1}_total'].append(None)
                log_dict[f'task{i+1}_individual'].append(None)
                log_dict[f'task{i+1}_shared'].append(None)
            if accuracies:
                log_dict[f'task{i+1}_accuracy'].append(accuracies[i])
            else:
                log_dict[f'task{i+1}_accuracy'].append(None)
            if task_shared_capacities:
                for k,v in task_shared_capacities[i].items():
                    log_dict[f'task{i+1}_shared_{k}'].append(v)
            else:
                for k in task_share_combinations(self.n_models):
                    log_dict[f'task{i+1}_shared_{k}'].append(None)

    def export_csv(self, path: str) -> None:
        self.log = open(self.log_path, 'r', encoding='UTF8')

        counter = 1
        base_accuracies, base_latency = self.original_model_parser()
        self.n_models = len(base_accuracies)

        log_dict = {}
        log_dict['total_capacity'] = []
        log_dict['latency'] = []
        log_dict['compile_time'] = []
        for i in range(self.n_models):
            log_dict[f'task{i+1}_total'] = []
            log_dict[f'task{i+1}_individual'] = []
            log_dict[f'task{i+1}_shared'] = []
            log_dict[f'task{i+1}_accuracy'] = []
            for k in task_share_combinations(self.n_models):
                log_dict[f'task{i+1}_shared_{k}'] = []
        self.append_base_info(log_dict, base_accuracies, base_latency)

        while line := self.log.readline():
            if line == '\n' or line.strip().endswith("--"):
                break
            else:
                node1_info, node2_info, total_capacity, task_capacities, task_shared_capacities, accuracies, latency, compile_time = self.opt_iteration_parser()
                self.append_opt_info(log_dict, node1_info, node2_info, total_capacity, task_capacities, task_shared_capacities, accuracies, latency, compile_time)
                counter += 1
        
        dataframe = pd.DataFrame(log_dict)
        dataframe = dataframe.replace(to_replace='None', value=np.nan).dropna()
        dataframe.to_csv(path, index=False)
        
        self.log.close()


if __name__ == "__main__":
    parser = LogParser('tmp/opt2.log')
    parser.export_csv('tmp/opt2.csv')
