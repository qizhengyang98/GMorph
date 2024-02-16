from typing import List, Dict, Callable

import copy
import timeit
from functools import partial

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import transformers

import random

from metamorph.graph.abs_graph import Graph
from metamorph.graph.cmp_graph import ComputeGraph
from metamorph.compiler.policy import Policy
from metamorph.metrics.testing_utils import test_accuracy, test_latency, test_latency_cuda
from metamorph.config.config import get_log_file_loc
from metamorph.misc.types import Result, FinetuneLevel
from metamorph.metrics.testing_utils import HiddenPrints

import logging
import time
import csv

class MetaMorph:
    BASIC_OPS = (
        nn.Conv2d,
        nn.ReLU,
        nn.BatchNorm2d,
        nn.Linear,
        nn.Dropout,
        nn.MaxPool2d,
        nn.AdaptiveAvgPool2d,
        nn.Flatten
    )

    def __init__(
            self,
            models: List[nn.Module],
            optimizer: optim.Optimizer,
            optimizer_lr: float,
            input_size: torch.Tensor,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            f_accuracy: Callable,
            f_latency: Callable,
            fine_tune_epochs=2,
            max_epoch=50,
            enable_fine_tune_early_stop=True,
            fine_tune_early_stop_check_epoch=10,
            custom_unmergeable_type=[],
            use_transformer=False,
            device='cuda'
        ) -> None:
        # External Definitions
        self.device = device
        self.models = []
        self.optimizer = optimizer
        self.optimizer_lr = optimizer_lr
        self.input_size = input_size
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dummy_input = torch.ones(input_size).to(self.device)
        self.fine_tune_epochs = fine_tune_epochs
        self.max_epoch = max_epoch
        self.enable_fine_tune_early_stop = enable_fine_tune_early_stop
        self.fine_tune_early_stop_check_epoch = fine_tune_early_stop_check_epoch

        # for BERT or tasks which takes text embeddings as input
        if self.dummy_input.dim() == 2:
            self.dummy_input = self.dummy_input.int()

        if custom_unmergeable_type:
            Graph.UNMERGEABLE_NODES += custom_unmergeable_type

        # init model list
        self.use_transformer = use_transformer
        self.init_models(models)

        # Latency Testing Parameters
        self.timing_number = 30
        self.timing_repeat = 30
        self.f_accuracy = partial(
            test_accuracy,
            f_test=f_accuracy,
            test_loader=test_loader,
            device=self.device
        )
        self.f_latency = partial(
            f_latency,
            # test_latency_cuda,
            dummy_input=self.dummy_input,
            timing_repeat=self.timing_repeat,
            timing_number=self.timing_number
        )

        # Abstract Graph of the Original Model
        self.original_graph = Graph(self.dummy_input, self.models, use_transformer=self.use_transformer, device=self.device)
        # self.original_models = [model.to(self.device) for model in models]
        self.unload_models(models)

        self.result = None
        self.compile_start_time, self.compile_end_time = None, None

        logging.basicConfig(filename=get_log_file_loc(), format='%(asctime)s %(levelname)-4s %(message)s',
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

        logging.info('The compiler starts successfully! ')
        if self.device == 'cpu':
            logging.info(f'Device info: cpu')
        else:
            logging.info(f'Device info: {torch.cuda.get_device_name(self.device)}')
        logging.info(f'Hyper Params: input size {self.input_size}, batch size {self.train_loader.batch_size}, lr {self.optimizer_lr}')
        logging.info(f'Settings: fine-tune epoch {self.fine_tune_epochs}, max iteration {self.max_epoch}, early_stop_check_epoch {self.fine_tune_early_stop_check_epoch}, enable_eary_stop {self.enable_fine_tune_early_stop}')
        logging.info(f'Original Graph:\n {self.original_graph}')
        total_capacity, branch_capacities = self.original_graph.capacity()
        logging.info(f'total capacity: {total_capacity}')
        logging.info('task specific capacity: ')
        for i, cap in enumerate(branch_capacities):
            logging.info(cap)  
        
        self.export_tracing_record = False

    def parse_model(self, model: nn.Module) -> List[nn.Module]:
        res = []
        if self.use_transformer:
            for layer in model.children():
                if type(layer) in MetaMorph.BASIC_OPS:
                    res.append(layer)
                elif isinstance(layer, nn.Sequential):
                    res.extend(self.parse_model(layer))
                elif isinstance(layer, transformers.models.vit.modeling_vit.ViTModel):
                    res.extend(self.parse_model(layer))
                elif isinstance(layer, transformers.models.vit.modeling_vit.ViTEncoder):
                    res.extend(self.parse_model(layer))
                elif isinstance(layer, transformers.models.bert.modeling_bert.BertModel):
                    res.extend(self.parse_model(layer))
                elif isinstance(layer, transformers.models.bert.modeling_bert.BertEncoder):
                    res.extend(self.parse_model(layer))
                elif isinstance(layer, torch.nn.modules.container.ModuleList):
                    res.extend(self.parse_model(layer))
                else:
                    res.append(layer)
        else:
            for layer in model.children():
                if type(layer) in MetaMorph.BASIC_OPS:
                    res.append(layer)
                elif isinstance(layer, nn.Sequential):
                    res.extend(self.parse_model(layer))
                else:
                    res.append(layer)
        return res

    def init_models(self, models: List[nn.Module]):
        for model in models:
            model = model.to(self.device)
            self.models.append(self.parse_model(model))

    def unload_models(self, models: List[nn.Module]):
        for model in models:
            model = model.cpu()

    def optimize(self, policy: Policy) -> Graph:
        logging.info("-------------------------- Optimization Begins ----------------------------------")
        torch.cuda.synchronize()
        self.compile_start_time = time.time()
        
        epoch = 0
        self.result = Result(self.original_graph, policy.latency_baseline, None)

        while epoch < self.max_epoch and not policy.early_stop():
            print("---------- Epoch: {}/{}, current best latency: {}".format(epoch+1, self.max_epoch, self.result[1]))
            logging.info(f'Current iteration: {epoch+1}/{self.max_epoch}, Current best latency: {self.result[1]}')
            
            candidate, cand_model = policy.step(epoch)
            if candidate:
                cand_graph, cand_latency = candidate 
                if not self.result or cand_latency < self.result.latency:
                    logging.info('Change the best result to the current result ')
                    self.result = Result(cand_graph, cand_latency, cand_model)
            epoch += 1

            torch.cuda.synchronize()
            self.compile_end_time = time.time()
            logging.info(f'Compiling time: {self.compile_end_time - self.compile_start_time}')
            logging.info('-------------------------------------------------')

        logging.info("-------------------------- Optimization Ends ------------------------------------")
        logging.info(f'The Best Graph:\n {self.result.graph}')
        # if self.result.cmp_graph is not None:
        #     with HiddenPrints():
        #         best_acc = self.f_accuracy(self.result.cmp_graph.to(self.device))
        #     print_best_acc = ''
        #     for i, acc in enumerate(best_acc):
        #         print_best_acc += f"net{i+1}: {acc.item()*100}%   "
        #     logging.info(f'Acc of The Best Graph: {print_best_acc}')
        logging.info(f'The Best Latency: {self.result.latency}')
        logging.info(f'Total Compiling Time: {self.compile_end_time - self.compile_start_time}')
        logging.info(f'The number of generated graph: {len(policy.exist_graphs)}')
        return self.result
    
    def fine_tune(self, cmp_graph: ComputeGraph, accuracy_baseline: torch.Tensor, accuracy_tolerence: float):
        # base_result = [0 for _ in range(len(self.models))]
        task_loss = [nn.L1Loss() for _ in range(len(self.models))]
        trace_init_loss = [999 for _ in range(len(self.models))]
        trace_avg_loss = [[0 for _ in range(len(self.models))] for _ in range(self.fine_tune_epochs)]
        trace_after_loss = [[0 for _ in range(len(self.models))] for _ in range(self.fine_tune_epochs)]
        trace_val_acc_drop = [[0 for _ in range(len(self.models))] for _ in range(self.fine_tune_epochs)]
        
        optimizer = self.optimizer(cmp_graph.params, lr=self.optimizer_lr)

        for e in range(self.fine_tune_epochs):
            cmp_graph.train()

            loop = tqdm(
                enumerate(self.train_loader), total=len(self.train_loader), leave=False
            )

            # control the early stopping condition
            # if self.enable_fine_tune_early_stop:
            if e % self.fine_tune_early_stop_check_epoch == 0:
                with HiddenPrints():
                    cur_accuracy = self.f_accuracy(cmp_graph)
                # get the avg accuracy drop between tasks
                # acc_delta = sum(accuracy_baseline - cur_accuracy) / len(accuracy_baseline)
                # get the max accuracy drop between tasks

                task_acc_drop = accuracy_baseline - cur_accuracy
                acc_delta = torch.max(task_acc_drop)
                # # trace the accuracy on val data
                trace_val_acc_drop[e] = [t.item() for t in task_acc_drop]
                if acc_delta <= accuracy_tolerence and self.enable_fine_tune_early_stop:
                    print('Accuracy drop: {} meets the threshold, early stop!'.format(acc_delta))
                    logging.info(f'Acc drop meets the threshold, early stop as epoch {e+1}/{self.fine_tune_epochs} !')
                    break

            count_batch = 0
            for _, (x_in, dl_id, y_out) in loop:
                if isinstance(x_in, Dict):
                    x_in = {k:v.to(self.device) for k,v in x_in.items()}
                else:
                    x_in = x_in.to(self.device)
                y_out = [y_out_.to(self.device) for y_out_ in y_out]
                loss = 0
                shared_result = cmp_graph(x_in)
                for i, model in enumerate(self.models):
                    # base_result = model(x_in)
                    tmp_loss = task_loss[i](shared_result[i], y_out[i])
                    loss += tmp_loss
                    # trace the loss
                    if trace_init_loss[i] == 999:
                        trace_init_loss[i] = tmp_loss.item()
                    trace_after_loss[e][i] = tmp_loss.item()
                    trace_avg_loss[e][i] += tmp_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loop.set_description(f"Epoch [{e+1}/{self.fine_tune_epochs}]")
                loop.set_postfix(loss = loss.item())
                count_batch += 1

            trace_avg_loss[e] = [t/count_batch for t in trace_avg_loss[e]]
        trace_after_loss.insert(0, trace_init_loss)
        trace_avg_loss.insert(0, trace_init_loss)
        if self.export_tracing_record:
            self.export_acc_drop_and_loss(trace_after_loss, trace_avg_loss, trace_val_acc_drop)

    # fine tune with learning curve extrapolation to early terminate
    def fine_tune_w_lc_extr(self, cmp_graph: ComputeGraph, accuracy_baseline: torch.Tensor, accuracy_tolerence: float):
        # base_result = [0 for _ in range(len(self.models))]
        task_loss = [nn.L1Loss() for _ in range(len(self.models))]
        trace_val_acc_drop = []
        
        optimizer = self.optimizer(cmp_graph.params, lr=self.optimizer_lr)

        force_val_acc = False

        for e in range(self.fine_tune_epochs):
            cmp_graph.train()

            loop = tqdm(
                enumerate(self.train_loader), total=len(self.train_loader), leave=False
            )

            # control the early stopping condition
            # if self.enable_fine_tune_early_stop:
            if e % self.fine_tune_early_stop_check_epoch == 0 or force_val_acc:
                with HiddenPrints():
                    cur_accuracy = self.f_accuracy(cmp_graph)

                task_acc_drop = accuracy_baseline - cur_accuracy
                acc_delta = torch.max(task_acc_drop)
                # # trace the accuracy on val data
                if not trace_val_acc_drop or acc_delta <= trace_val_acc_drop[-1] + accuracy_tolerence:
                    trace_val_acc_drop.append(acc_delta)
                    force_val_acc = False
                else:
                    force_val_acc = True
                # early stop condition
                if acc_delta <= accuracy_tolerence and self.enable_fine_tune_early_stop:
                    print('Accuracy drop: {} meets the threshold, early stop!'.format(acc_delta))
                    logging.info(f'Acc drop meets the threshold, early stop as epoch {e+1}/{self.fine_tune_epochs} !')
                    break
                # early ternimation with learning curve extrapolation
                if len(trace_val_acc_drop) >= 4:
                    _, est_acc_drop = self.est_final_acc_drop(trace_val_acc_drop, e, accuracy_tolerence)
                    if est_acc_drop > accuracy_tolerence:
                    # if est_acc_drop > accuracy_tolerence + 0.01: # add 0.01 to avoid deviation
                        print(f'Predicted Accuracy drop at epoch {e+1}: {est_acc_drop} cannot meet the threshold, terminate!')
                        logging.info(f'Predicted Accuracy drop at epoch {e+1}: {est_acc_drop} cannot meet the threshold, terminate!')
                        break

            count_batch = 0
            for _, (x_in, dl_id, y_out) in loop:
                if isinstance(x_in, Dict):
                    x_in = {k:v.to(self.device) for k,v in x_in.items()}
                else:
                    x_in = x_in.to(self.device)
                y_out = [y_out_.to(self.device) for y_out_ in y_out]
                loss = 0
                shared_result = cmp_graph(x_in)
                for i, model in enumerate(self.models):
                    # base_result = model(x_in)
                    tmp_loss = task_loss[i](shared_result[i], y_out[i])
                    loss += tmp_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loop.set_description(f"Epoch [{e+1}/{self.fine_tune_epochs}]")
                loop.set_postfix(loss = loss.item())
                count_batch += 1

    def est_final_acc_drop(self, acc_list, cur_epoch, target):
        est_n = -1
        num_base_point = 4
        acc_samples = np.array(acc_list)[-num_base_point:]
        n_step = int(np.floor(self.fine_tune_epochs / self.fine_tune_early_stop_check_epoch))
        cur_step = int(np.floor(cur_epoch / self.fine_tune_early_stop_check_epoch))

        if acc_samples[3] == 0:
            return est_n, 0.009
        
        x01 = np.log(np.abs(acc_samples[0]-acc_samples[1]))
        x12 = np.log(np.abs(acc_samples[1]-acc_samples[2]))
        x23 = np.log(np.abs(acc_samples[2]-acc_samples[3]))
        alpha = (x23 - x12) / (x12 - x01)

        cur1, cur2 = x12, x23
        cur_x = acc_samples[3]
        for i in range(cur_step, n_step+1):
            # print(f'cur_step: {cur_step}, pred acc {cur_x}')
            next = alpha * (cur2 - cur1) + cur2
            cur_x -= np.exp(next)
            if cur_x <= target:
                est_n = i
                break
        return est_n, cur_x

    def record_acc_drop_and_loss(self, file_loc, file_name):
        self.export_loc_filename = f"{file_loc}/{file_name}"
        self.export_tracing_record = True

    def export_acc_drop_and_loss(self, after_loss, avg_loss, val_acc_drop):
        for i in range(len(self.models)):
            with open(f'{self.export_loc_filename}_after_loss_task{i+1}.csv', 'a+', newline='') as f1:
                write = csv.writer(f1)
                write.writerow([afl[i] for afl in after_loss])
            with open(f'{self.export_loc_filename}_avg_loss_task{i+1}.csv', 'a+', newline='') as f2:
                write = csv.writer(f2)
                write.writerow([avl[i] for avl in avg_loss])
            with open(f'{self.export_loc_filename}_acc_drop_task{i+1}.csv', 'a+', newline='') as f3:
                write = csv.writer(f3)
                write.writerow([vad[i] for vad in val_acc_drop])

    def subgraph_finetuning_sanity_check(self, n_connect=1, level=1, load_weight=True):
        logging.info("------------------ Subgraph Finetuning Sanity Check ... ----------------------")
        merged_graph_sub, merged_graph_all = self.get_merged_graph_sanity_check(n_connect, level)
        logging.info(f'Merged Graph:\n {merged_graph_sub}')
        tmp_early_stop_flag = self.fine_tune_early_stop_check_epoch
        self.fine_tune_early_stop_check_epoch = None

        cmp_graph = {'Sub-graph':merged_graph_sub, 'Entire-graph':merged_graph_all}
        for g in cmp_graph:
            logging.info(f"{g} finetuning ...")
            merged_cmp_graph = ComputeGraph(cmp_graph[g], self.models, device=self.device, use_transformer=self.use_transformer, load_weight=load_weight)
            merged_cmp_graph.check_requires_grad()
            torch.cuda.synchronize()
            finetune_start_time = time.time()
            fake_acc_baseline, fake_acc_thres = torch.ones(len(self.models)), 0.0
            self.fine_tune(merged_cmp_graph, fake_acc_baseline, fake_acc_thres)
            torch.cuda.synchronize()
            finetune_end_time = time.time()
            logging.info(f'{g} Total finetune Time: {finetune_end_time - finetune_start_time}')

        self.fine_tune_early_stop_check_epoch = tmp_early_stop_flag
        logging.info('-------------------------------------------------')

    def get_merged_graph_sanity_check(self, n_connect=1, level=1) -> Graph:
        merged_graph_sub = copy.deepcopy(self.original_graph)
        merged_graph_all = copy.deepcopy(self.original_graph)
        for _ in range(n_connect):
            merged_graph_sub.set_req_grad(FinetuneLevel.SUBGRAPH)
            merged_graph_sub.merge_sanity_check(num_models=len(self.models), level=level)
            merged_graph_sub.build_mergeable_nodes()
            merged_graph_all.set_req_grad(FinetuneLevel.FULLGRAPH)
            merged_graph_all.merge_sanity_check(num_models=len(self.models), level=level)
            merged_graph_all.build_mergeable_nodes()
        return merged_graph_sub, merged_graph_all