import torch

from metamorph.misc.test_UTK import read_data, test, test_age_gen, test_age_eth, test_gen_eth
from metamorph.misc.multiNN import TridentNN
from metamorph.misc.multiNN_vgg13 import TridentNN_vgg13
from metamorph.compiler.compiler import MetaMorph
from metamorph.metrics.testing_utils import test_latency, test_latency_cuda
from metamorph.compiler.policy import SimulatedAnnealingPolicy, ManualSimulatedAnnealingPolicy, IterativePolicy, FilterBasedSimulatedAnnealingPolicy, LCBasedSimulatedAnnealingPolicy, RandomSamplingPolicy
from metamorph.misc.types import FinetuneLevel
from metamorph.data.dataloader import DatasetSampler
from metamorph.config import set_log_file_loc

import argparse

seed = 10
torch.manual_seed(seed)

def load_model(age_num, gen_num, eth_num):
    # tridentNN = TridentNN(age_num, gen_num, eth_num)
    tridentNN = TridentNN_vgg13(age_num, gen_num, eth_num)
    tridentNN.load_state_dict(torch.load('../model/toy_vgg13.pt', map_location='cpu'))
    ageNet, genNet, ethNet = tridentNN.ageNN, tridentNN.genNN, tridentNN.ethNN
    return ageNet, genNet, ethNet

def load_2model(age_num, gen_num, eth_num, task=['age', 'gen']):
    if 'age' in task and 'gen' in task:
        return [load_model(age_num, gen_num, eth_num)[0]] + [load_model(age_num, gen_num, eth_num)[1]]
    elif 'age' in task and 'eth' in task:
        return [load_model(age_num, gen_num, eth_num)[0]] + [load_model(age_num, gen_num, eth_num)[2]]
    elif 'gen' in task and 'eth' in task:
        return [load_model(age_num, gen_num, eth_num)[1]] + [load_model(age_num, gen_num, eth_num)[2]]
    else:
        raise ValueError("Task must be age, gen, or eth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_select', help="policy mode: ['random', 'iterative', 'manual', 'SimulatedAnnealing', 'FilterBased', 'LCBased']", type=str)
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
    parser.add_argument('--early_stop_check_epochs', help="check early stop condition every n epochs", type=int, default=2)
    parser.add_argument('--enable_filtering_rules', help="whether to filter graphs by rules", action='store_true')
    parser.set_defaults(enable_filtering_rules=False)
    parser.add_argument('--max_iteration', help="The maximum iteration of optimization", type=int, default=200)
    parser.add_argument('--batch_size', help="The batch size for fine-tuning", type=int, default=256)
    parser.add_argument('--num_workers', help="The number of workers", type=int, default=4)

    args = parser.parse_args()
    
    if_save_model = True
    set_log_file_loc(f"../../results/log/{args.log_name}.log")
    save_history = f"../../results/history/{args.log_name}.json"
    save_analyze = "../../results/analyze/opt6.json"
    save_model = f"../../results/model/{args.log_name}.pt"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    bs = args.batch_size

    trace_loss_acc_loc = "../../results/loss"
    trace_loss_acc_file = args.log_name

    train_loader, test_loader, n_class = read_data(batch_size=bs)
    for i,j in train_loader:
        sample_input = torch.ones(i[:1].shape)
        break

    tasks = ['age', 'gen', 'eth']
    age_num, gen_num, eth_num = n_class['age_num'], n_class['gen_num'], n_class['eth_num']
    if tasks == ['age', 'gen']:
        models = load_2model(age_num, gen_num, eth_num, tasks)
        test_func = test_age_gen
    elif tasks == ['age', 'eth']:
        models = load_2model(age_num, gen_num, eth_num, tasks)
        test_func = test_age_eth
    elif tasks == ['gen', 'eth']:
        models = load_2model(age_num, gen_num, eth_num, tasks)
        test_func = test_gen_eth
    else:
        models = load_model(age_num, gen_num, eth_num)
        test_func = test

    ds_samples = DatasetSampler(
        [train_loader.dataset],
        models,
        device,
        # keep_sizes = [1024],
        keep_sizes = [len(train_loader.dataset)],
        batch_size= bs,
        rand_seed=seed
    )
    train_loader = torch.utils.data.DataLoader(ds_samples, batch_size=bs, shuffle=False, **kwargs)
    print(f"the size of training set is {len(ds_samples)}, the size of testing set is {len(test_loader.dataset)}\n")

    optimizer = torch.optim.Adam
    compiler = MetaMorph(
        models=models, optimizer=optimizer, optimizer_lr=0.0005,
        input_size=sample_input.shape, train_loader=train_loader, test_loader=test_loader,
        f_accuracy=test_func, f_latency=test_latency, 
        fine_tune_epochs=args.fine_tune_epochs, max_epoch=args.max_iteration, device=device,
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
            accuracy_tolerence = args.acc_drop_thres, initial_temp = 100, alpha=args.alpha, fine_tune_level=fine_tune_level, 
            device=compiler.device
        )
        policy.load_history_from_json(save_history)
    elif args.policy_select=='iterative':
        policy = IterativePolicy(
            base_graph=compiler.original_graph,
            models=compiler.models,
            f_finetune=compiler.fine_tune, f_latency=compiler.f_latency, f_accuracy=compiler.f_accuracy, load_weight=args.load_weight,
            accuracy_tolerence = args.acc_drop_thres, fine_tune_level=fine_tune_level, 
            device=compiler.device
        )
    elif args.policy_select=='FilterBased':
        policy = FilterBasedSimulatedAnnealingPolicy(
            base_graph=compiler.original_graph,
            models=compiler.models,
            f_finetune=compiler.fine_tune, f_latency=compiler.f_latency, f_accuracy=compiler.f_accuracy, load_weight=args.load_weight,
            filtering_model_type="xgbc", num_starting_topo=32,
            accuracy_tolerence = args.acc_drop_thres, initial_temp = 100, alpha=args.alpha, fine_tune_level=fine_tune_level,
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
    elif args.policy_select=='random':
        policy = RandomSamplingPolicy(
            base_graph=compiler.original_graph,
            models=compiler.models,
            f_finetune=compiler.fine_tune, f_latency=compiler.f_latency, f_accuracy=compiler.f_accuracy, load_weight=args.load_weight,
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
    compiler.record_acc_drop_and_loss(trace_loss_acc_loc, trace_loss_acc_file)
    policy.record_history(save_history)

    best_result = compiler.optimize(policy)
    print("Optimal Graph: \n", best_result.graph)
    print("Optimal Latency: ", best_result.latency)

    best_model = best_result.cmp_graph

    if args.policy_select=='SimulatedAnnealing':
        policy.export_merge_history(save_history)
        policy.save_analyze(save_analyze)
    if best_model is not None and if_save_model:
        torch.save(best_model, save_model)
        test(test_loader, best_model, device)


if __name__ == "__main__":
    main()
