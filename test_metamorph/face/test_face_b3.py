import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('facial_expression')
sys.path.append('facial_age_gender')

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from facial_expression.VGG_Face_torch import VGG_emotionNet, VGG_emotionNet_13
from facial_age_gender.VGG_Face_torch import VGG_ageNet, VGG_genderNet, VGG_genderNet_11
from test_func import test_multi_acc

from metamorph.compiler.compiler import MetaMorph
from metamorph.metrics.testing_utils import test_accuracy, test_latency
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
parser.add_argument('--batch_size', help="The batch size for fine-tuning", type=int, default=64)
parser.add_argument('--num_workers', help="The number of workers", type=int, default=4)

args = parser.parse_args()

seed = 10
torch.manual_seed(seed)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAMPLE_INPUT = torch.rand(1,3,224,224).to(DEVICE)
kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
bs = args.batch_size
if_save_model = True

print(f"Running on {DEVICE}, input shape is {SAMPLE_INPUT.shape}, batch size is {bs}\n")
set_log_file_loc(f"../../results/log/{args.log_name}.log")
save_model = f"../../results/models/{args.log_name}.pt"
save_history = f"../../results/history/{args.log_name}.json"
save_analyze = f"../../results/analyze/{args.log_name}.json"

trace_loss_acc_loc = "../../results/loss"
trace_loss_acc_file = args.log_name

# emotionNet + FER2013
emo_transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.Resize(256),
                                 transforms.RandomCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                                ])
emo_transform_test  = transforms.Compose([transforms.Resize(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                                ])
emo_train_data = torchvision.datasets.ImageFolder('../../datasets/fer2013/train',transform=emo_transform_train)
emo_test_data = torchvision.datasets.ImageFolder('../../datasets/fer2013/test',transform=emo_transform_test)
emo_test_loader = torch.utils.data.DataLoader(emo_test_data, batch_size=bs, shuffle=False, **kwargs)

emotionNet = VGG_emotionNet_13()
emotionNet.load_state_dict(torch.load('pre_models/EmotionNet_vgg13.model', map_location='cpu'))
emotionNet = emotionNet.model

emotionNet = emotionNet.eval()
print("emotionNet is loaded successfully!")

# ageNet + Adience
age_transform_test  = transforms.Compose([transforms.Resize(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                                ])
age_data = torchvision.datasets.ImageFolder('../../datasets/adience/age',transform=age_transform_test)
age_train_indices, age_test_indices = train_test_split(list(range(len(age_data.targets))), test_size=0.2, stratify=age_data.targets, random_state=10)
age_train_data = torch.utils.data.Subset(age_data, age_train_indices)
age_test_data = torch.utils.data.Subset(age_data, age_test_indices)
age_test_loader = torch.utils.data.DataLoader(age_test_data, batch_size=bs, shuffle=False, **kwargs)

ageNet = VGG_ageNet()
ageNet.load_state_dict(torch.load('pre_models/ageNet.model', map_location='cpu'))
ageNet = ageNet.eval()
print("ageNet is loaded successfully!")

# genderNet + Adience
gen_transform_test  = transforms.Compose([transforms.Resize(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                                ])
gender_data = torchvision.datasets.ImageFolder('../../datasets/adience/gender',transform=gen_transform_test)
gen_train_indices, gen_test_indices = train_test_split(list(range(len(gender_data.targets))), test_size=0.2, stratify=gender_data.targets, random_state=10)
gen_train_data = torch.utils.data.Subset(gender_data, gen_train_indices)
gen_test_data = torch.utils.data.Subset(gender_data, gen_test_indices)
gen_test_loader = torch.utils.data.DataLoader(gen_test_data, batch_size=bs, shuffle=False, **kwargs)

genderNet = VGG_genderNet_11()
genderNet.load_state_dict(torch.load('pre_models/genderNet_vgg11.model', map_location='cpu'))
genderNet = genderNet.model

genderNet = genderNet.eval()
print("genderNet is loaded successfully!")

# list of models
MODELS = [emotionNet, ageNet, genderNet]

# dataloader
ds_samples = DatasetSampler(
        [emo_train_data, age_train_data, gen_train_data],
        MODELS,
        DEVICE,
        # [30,30,30]
        keep_sizes=[10000, 5000, 5000],
        batch_size=bs,
        rand_seed=seed
    )
samples_dataloader = torch.utils.data.DataLoader(ds_samples, batch_size=bs, shuffle=True, **kwargs)
test_loader_list = [emo_test_loader, age_test_loader, gen_test_loader]
print("traning samples loaded successfully!")
test_data_size = len(emo_test_data) + len(age_test_data) + len(gen_test_data)
print(f"the size of training set is {len(ds_samples)}, the size of testing set is {test_data_size}\n")

# torch.cuda.empty_cache()


import time

# compiler settings
optimizer = torch.optim.Adam
compiler = MetaMorph(
    models=MODELS, optimizer=optimizer, optimizer_lr=0.001,
    input_size=SAMPLE_INPUT.shape, train_loader=samples_dataloader, test_loader=test_loader_list,
    f_accuracy=test_multi_acc, f_latency=test_latency,
    fine_tune_epochs=args.fine_tune_epochs, max_epoch=args.max_iteration, device=DEVICE,
    enable_fine_tune_early_stop=args.finetune_early_stop, fine_tune_early_stop_check_epoch=args.early_stop_check_epochs
)

if args.sub_graph_finetune:
    fine_tune_level = FinetuneLevel.SUBGRAPH
else:
    fine_tune_level = FinetuneLevel.FULLGRAPH
   
if args.policy_select=='manual':
    policy = ManualSimulatedAnnealingPolicy(
        base_graph=compiler.original_graph,
        models=compiler.models,
        f_finetune=compiler.fine_tune, f_latency=compiler.f_latency, f_accuracy=compiler.f_accuracy, load_weight=args.load_weight,
        accuracy_tolerence = args.acc_drop_thres, initial_temp = 100, alpha=args.alpha, fine_tune_level=FinetuneLevel.FULLGRAPH, 
        device=compiler.device
    )
    policy.load_history_from_json(save_history)
elif args.policy_select=='iterative':
    policy = IterativePolicy(
        base_graph=compiler.original_graph,
        models=compiler.models,
        f_finetune=compiler.fine_tune, f_latency=compiler.f_latency, f_accuracy=compiler.f_accuracy, load_weight=args.load_weight,
        accuracy_tolerence = args.acc_drop_thres, fine_tune_level=FinetuneLevel.FULLGRAPH, 
        device=compiler.device
    )
elif args.policy_select=='FilterBased':
    policy = FilterBasedSimulatedAnnealingPolicy(
        base_graph=compiler.original_graph,
        models=compiler.models,
        f_finetune=compiler.fine_tune, f_latency=compiler.f_latency, f_accuracy=compiler.f_accuracy, load_weight=args.load_weight,
        filtering_model_type="xgbc", num_starting_topo=32,
        accuracy_tolerence = args.acc_drop_thres, initial_temp = 100, alpha=args.alpha, fine_tune_level=FinetuneLevel.FULLGRAPH,
        device=compiler.device
    )
elif args.policy_select=='LCBased':
    policy = LCBasedSimulatedAnnealingPolicy(
        base_graph=compiler.original_graph,
        models=compiler.models,
        f_finetune=compiler.fine_tune_w_lc_extr, f_latency=compiler.f_latency, f_accuracy=compiler.f_accuracy, load_weight=args.load_weight,
        enable_filtering_rules=args.enable_filtering_rules,
        accuracy_tolerence = args.acc_drop_thres, initial_temp = 100, alpha=args.alpha, fine_tune_level=fine_tune_level,
        device=compiler.device
    )
else:
    policy = SimulatedAnnealingPolicy(
        base_graph=compiler.original_graph,
        models=compiler.models,
        f_finetune=compiler.fine_tune, f_latency=compiler.f_latency, f_accuracy=compiler.f_accuracy, load_weight=args.load_weight,
        accuracy_tolerence = args.acc_drop_thres, initial_temp = 100, alpha=args.alpha, fine_tune_level=fine_tune_level,
        device=compiler.device
    )
    
# trace the training loss and acc drop on val data
# compiler.record_acc_drop_and_loss(trace_loss_acc_loc, trace_loss_acc_file)
# policy.record_history(save_history)
    
print("compiler is set! Start compiling ...\n")
print('---------------------------- Compiling ---------------------------------')

best_result = compiler.optimize(policy)

print('---------------------------- Evaluation ---------------------------------')
print("Optimal Graph: \n", best_result.graph)
print("Optimal Latency: ", best_result.latency)

# if args.policy_select=='SimulatedAnnealing':
#     policy.export_merge_history(save_history)
#     policy.save_analyze(save_analyze)

cmpGraph_opt = best_result.cmp_graph
print("Task Accuracy of optimized graph: ")

if cmpGraph_opt is not None and if_save_model:
    torch.save(cmpGraph_opt, save_model)
    test_accuracy(cmpGraph_opt, test_multi_acc, test_loader_list, DEVICE)