import numpy as np
import torch

import tvm, time
from tvm import relay, autotvm
from tvm.contrib import graph_executor
import argparse
import timeit

from tvm_build import get_network, tune_tasks, tune_and_evaluate
from benchmark import results_toymodel, results_scene, results_face16, results_face


parser = argparse.ArgumentParser()
parser.add_argument('--task', help="test task: ['toy', 'scene', 'face16', 'face']", type=str)
parser.add_argument('--model', help="test model: ['origin', 'merge_half_conv', 'merge_all_conv', 'SA', 'LC_wo_rule', 'LC_w_rule']", type=str)
parser.add_argument('--use_tune_config', help="with or without tuning config for inference", action='store_true')
parser.set_defaults(use_tune_config=False)
parser.add_argument('--enable_tune', help="do tuning", action='store_true')
parser.set_defaults(enable_tune=False)

args = parser.parse_args()

def test_tvm_latency(tvm_model, timing_number=30, timing_repeat=30):
    # GPU Warmup
    for _ in range(10):
        _ = tvm_model.run()

    latency = (
            np.array(timeit.Timer(lambda:tvm_model.run()).repeat(repeat=timing_repeat, number=timing_number))
            * 1000
            / timing_number
        )
    latency = {"mean": np.mean(latency), "median": np.median(latency), "std":np.std(latency)}

    print("Inference Latency: %s" % (latency))
    return latency['mean']

def test_tvm_latency_cuda(tvm_model, timing_repeat=30, timing_number=30):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((timing_repeat, timing_number))
    timing_mean, timing_std = None, None

    # GPU Warmup
    for _ in range(10):
        _ = tvm_model.run()
    
    while timing_mean is None or timing_std > timing_mean/10:
        # Measure Performance
        with torch.no_grad():
            for i in range(timing_repeat):
                for j in range(timing_number):
                    torch.manual_seed(i+j)
                    start.record()
                    _ = tvm_model.run()
                    end.record()
                    # Wait for GPU synchronization
                    torch.cuda.synchronize()
                    timings[i][j] = start.elapsed_time(end)
        
        timing_mean = np.mean(np.mean(timings, axis=1))
        timing_median = np.median(timings)
        timing_std = np.mean(np.std(timings, axis=1, ddof=1))
    
    latency = {"mean": timing_mean, "median": timing_median, "std": timing_std}
    print("Inference Latency: %s \n" % (latency))
    return latency['mean']


if __name__ == "__main__":
    torch.manual_seed(0)
    log_name = f"tvm_config/{args.task}_{args.model}.log"

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sample_input = torch.randn(1,3,224,224).to(DEVICE)
    print("Read config: ", log_name)
    
    if args.task == 'toy':
        model = results_toymodel[args.model].to(DEVICE)
        sample_input = torch.rand(1,1,48,48).to(DEVICE)
        arch = 'sm_61' #1080ti
    elif args.task == 'scene':
        model = results_scene[args.model].to(DEVICE)
        arch = 'sm_61' #1080ti
    elif args.task == 'face16':
        model = results_face16[args.model].to(DEVICE)
        arch = 'sm_75' #rtx8000
    elif args.task == 'face':
        model = results_face[args.model].to(DEVICE)
        arch = 'sm_75' #rtx8000
    else:
        print('Invalid task: task must be either toy or scene or face16 or face')

    traced_module = torch.jit.trace(model, sample_input).eval()
    input_name = 'input0'
    input_data = torch.randn(sample_input.shape)
    shape_list = [(input_name, sample_input.shape)]

    mod, params = get_network(traced_module, shape_list)
    print(1)

    # Auto Tuning
    if args.enable_tune:
        tune_and_evaluate(mod, params, sample_input.shape, log_name=log_name)

    # compile by tvm
    target = tvm.target.cuda(arch=arch)
    if args.use_tune_config:
        with autotvm.apply_history_best(log_name):
            with tvm.transform.PassContext(opt_level=3):
                lib = lib = relay.build(mod, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
    dev = tvm.device(str(target), 0)
    tvm_model = graph_executor.GraphModule(lib["default"](dev))  
    print(2)

    # test inference time
    tvm_model.set_input(input_name, input_data)
    tvm_model.run()
    
    print(f'Task: {args.task}, Model: {args.model}, Use_Config: {args.use_tune_config}')
    test_tvm_latency(tvm_model)
    test_tvm_latency_cuda(tvm_model)
    
    