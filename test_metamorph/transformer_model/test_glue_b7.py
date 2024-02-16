import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Tuple, Dict
import time
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.models as models
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

# from local files
from run_bert_glue import preprocess_function, collate_fn_sampler, load_data, load_data_sampler
from test_func import test_cola, test_sst2, test_multi_cola_sst2

# from GMorph package
from metamorph.compiler.compiler import MetaMorph
from metamorph.metrics.testing_utils import test_accuracy, test_latency, test_latency_cuda
from metamorph.compiler.policy import SimulatedAnnealingPolicy, ManualSimulatedAnnealingPolicy, IterativePolicy, FilterBasedSimulatedAnnealingPolicy, LCBasedSimulatedAnnealingPolicy
from metamorph.data.dataloader import DatasetSampler
from metamorph.config import set_log_file_loc
from metamorph.misc.types import FinetuneLevel

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--policy_select', help="policy mode: ['iterative', 'manual', 'SimulatedAnnealing', 'FilterBased', 'LCBased']", type=str)
parser.add_argument('--log_name', help="file name of the log output", type=str)
parser.add_argument('--load_weight', help="initialize model weight from existing models", action='store_true')
parser.set_defaults(load_weight=False)
parser.add_argument('--sub_graph_finetune', help="fine tune the merged sub-graph only", action='store_true')
parser.set_defaults(sub_graph_finetune=False)
parser.add_argument('--finetune_early_stop', help="enable early stop in finetune", action='store_true')
parser.set_defaults(finetune_early_stop=False)
parser.add_argument('--acc_drop_thres', help="Threshold of accuracy drop", type=float, default=0.01)
parser.add_argument('--alpha', help="The arg alpha to update temperature in SA", type=float, default=0.99)
parser.add_argument('--fine_tune_epochs', help="The number of epoch to finetune", type=int, default=40)
parser.add_argument('--early_stop_check_epochs', help="check early stop condition every n epochs", type=int, default=5)
parser.add_argument('--enable_filtering_rules', help="whether to filter graphs by rules", action='store_true')
parser.set_defaults(enable_filtering_rules=False)
parser.add_argument('--max_iteration', help="The maximum iteration of optimization", type=int, default=200)
parser.add_argument('--batch_size', help="The batch size for fine-tuning", type=int, default=32)
parser.add_argument('--num_workers', help="The number of workers", type=int, default=4)

args = parser.parse_args()
seed = 10
torch.manual_seed(seed)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAMPLE_INPUT = torch.ones((1,128), dtype=torch.int).to(DEVICE)
kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
bs = args.batch_size
if_save_model = False

root_folder = "../../"

print(f"Running on {DEVICE}, input shape is {SAMPLE_INPUT.shape}, batch size is {bs}\n")
set_log_file_loc(root_folder + f"results/log/{args.log_name}.log")
save_model = root_folder + f"results/models/{args.log_name}.pt"
save_history = root_folder + f"results/history/{args.log_name}.json"
save_analyze = root_folder + f"results/analyze/{args.log_name}.json"

trace_loss_acc_loc = root_folder + "results/loss"
trace_loss_acc_file = args.log_name


# GLUE task CoLA
cola_model_path = Path("cola/checkpoint")
cola_train_loader, cola_test_loader = load_data('cola', kwargs=kwargs)
cola_train_data_sampler = load_data_sampler('cola')
# print(len(cola_train_loader.dataset), len(cola_test_loader.dataset))
cola_config = AutoConfig.from_pretrained(cola_model_path)
cola_model = AutoModelForSequenceClassification.from_pretrained(cola_model_path, config=cola_config,)
# # Test: Matthews correlation coefficient: 0.6033586694276752
# cola_acc = test_cola(cola_model, cola_test_loader, DEVICE)
# print(cola_model)
print("cola_model is loaded successfully!")

# GLUE task SST-2
sst2_model_path = Path("sst2/checkpoint")
sst2_train_loader, sst2_test_loader = load_data('sst2', kwargs=kwargs)
sst2_train_data_sampler = load_data_sampler('sst2')
# print(len(sst2_train_loader.dataset), len(sst2_test_loader.dataset))
sst2_config = AutoConfig.from_pretrained(sst2_model_path)
sst2_model = AutoModelForSequenceClassification.from_pretrained(sst2_model_path, config=sst2_config,)
# # Test: Accuracy: 0.9174311926605505
# sst2_acc = test_sst2(sst2_model, sst2_test_loader, DEVICE)
# print(sst2_model)
print("sst2_model is loaded successfully!")

MODELS = [cola_model.eval(), sst2_model.eval()]
DATA_SAMPLER = [cola_train_data_sampler, sst2_train_data_sampler]

ds_samples = DatasetSampler(
        DATA_SAMPLER,
        MODELS,
        DEVICE,
        keep_sizes = [5000, 5000],
        batch_size = bs,
        rand_seed = seed,
        use_transformer = True,
        glue_task = True,
        preprocess_func = preprocess_function
    )
samples_dataloader = torch.utils.data.DataLoader(ds_samples, batch_size=bs, shuffle=True, collate_fn=collate_fn_sampler, **kwargs)
test_loader_list = [cola_test_loader, sst2_test_loader]
print("training samples loaded successfully!")
test_data_size = len(cola_test_loader.dataset) + len(sst2_test_loader.dataset)
print(f"the size of training set is {len(ds_samples)}, the size of testing set is {test_data_size}\n")

# compiler settings
optimizer = torch.optim.Adamax

customized_unmergebale_type = [transformers.models.bert.modeling_bert.BertEmbeddings]

compiler = MetaMorph(
    models=MODELS, optimizer=optimizer, optimizer_lr=5e-5,
    input_size=SAMPLE_INPUT.shape, train_loader=samples_dataloader, test_loader=test_loader_list,
    f_accuracy=test_multi_cola_sst2, f_latency=test_latency, 
    fine_tune_epochs=args.fine_tune_epochs, max_epoch=args.max_iteration,
    custom_unmergeable_type=customized_unmergebale_type, use_transformer=True, device=DEVICE,
    enable_fine_tune_early_stop=args.finetune_early_stop, fine_tune_early_stop_check_epoch=args.early_stop_check_epochs
)

if args.sub_graph_finetune:
    fine_tune_level = FinetuneLevel.SUBGRAPH
else:
    fine_tune_level = FinetuneLevel.FULLGRAPH

if args.policy_select=='manual':
    pass
elif args.policy_select=='iterative':
    pass
elif args.policy_select=='FilterBased':
    pass
elif args.policy_select=='LCBased':
    policy = LCBasedSimulatedAnnealingPolicy(
        base_graph=compiler.original_graph,
        models=compiler.models,
        f_finetune=compiler.fine_tune_w_lc_extr, f_latency=compiler.f_latency, f_accuracy=compiler.f_accuracy, load_weight=args.load_weight,
        enable_filtering_rules=args.enable_filtering_rules,
        accuracy_tolerence = args.acc_drop_thres, initial_temp = 100, alpha=args.alpha, fine_tune_level=fine_tune_level,
        use_transformer=compiler.use_transformer, device=compiler.device
    )
else:
    policy = SimulatedAnnealingPolicy(
        base_graph=compiler.original_graph,
        models=compiler.models,
        f_finetune=compiler.fine_tune, f_latency=compiler.f_latency, f_accuracy=compiler.f_accuracy, load_weight=args.load_weight,
        accuracy_tolerence = args.acc_drop_thres, initial_temp = 100, alpha=args.alpha, fine_tune_level=fine_tune_level,
        use_transformer=compiler.use_transformer, device=compiler.device
    )

# trace the training loss and acc drop on val data
compiler.record_acc_drop_and_loss(trace_loss_acc_loc, trace_loss_acc_file)
# policy.record_history(save_history)

print("compiler is set! Start compiling ...\n") 
print('---------------------------- Compiling ---------------------------------')

best_result = compiler.optimize(policy)

print('---------------------------- Evaluation ---------------------------------')
print("Optimal Graph: \n", best_result.graph)
print("Optimal Latency: ", best_result.latency)

if args.policy_select=='SimulatedAnnealing':
    policy.export_merge_history(save_history)
    policy.save_analyze(save_analyze)

cmpGraph_opt = best_result.cmp_graph
print("Task Accuracy of optimized graph: ")

if cmpGraph_opt is not None and if_save_model:
    torch.save(cmpGraph_opt, save_model)
    test_accuracy(cmpGraph_opt, test_multi_cola_sst2, test_loader_list, DEVICE)