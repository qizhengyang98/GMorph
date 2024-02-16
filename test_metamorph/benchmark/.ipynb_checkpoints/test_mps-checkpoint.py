import torch.multiprocessing as mp
import torch.nn as nn
import torch
import numpy as np
import timeit
import argparse

from benchmark import results_toymodel, results_scene, results_face16, results_face
from benchmark_toy import toy_single_task
from benchmark_scene import scene_single_task1, scene_single_task2
from benchmark_face16 import face16_single_task

parser = argparse.ArgumentParser()
parser.add_argument('--task', help="test task: ['toy', 'scene', 'face16', 'face']", type=str)

args = parser.parse_args()


def test_exec(model, dummy_inputs):
    torch.manual_seed(0)
    with torch.no_grad():
        for inp in dummy_inputs:
            ret = model(inp)
    return ret

def test_exec_single(model, dummy_input):
    torch.manual_seed(0)
    with torch.no_grad():
        ret = model(dummy_input)
    return ret


import time
if __name__ == '__main__':
    
    torch.manual_seed(0)

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # dummy_inputs = [torch.rand(1, 3, 224, 224).to(DEVICE) for _ in range(1000)]
    dummy_inputs = [torch.rand(1, 1, 48, 48).to(DEVICE) for _ in range(1000)]
    
    # if args.task == 'toy':
    #     models = [toy_single_task(num_output=10).eval().to(DEVICE), toy_single_task(num_output=2).eval().to(DEVICE), toy_single_task(num_output=3).eval().to(DEVICE)]
    #     dummy_inputs = [torch.rand(1, 1, 48, 48).to(DEVICE) for _ in range(1000)]
    # elif args.task == 'scene':
    #     models = [scene_single_task1().eval().to(DEVICE), scene_single_task2().eval().to(DEVICE)]
    # elif args.task == 'face16':
    #     models = [face16_single_task(num_output=7).eval().to(DEVICE), face16_single_task(num_output=8).eval().to(DEVICE), face16_single_task(num_output=2).eval().to(DEVICE)]
    # elif args.task == 'face':
    #     models = [results_face[args.model].to(DEVICE)]
    # else:
    #     print('Invalid task: task must be either toy or scene or face16 or face')
        
    models = [torch.jit.load('trt_toy_single_task1.ts')]
        
    print(1)
        
    num_processes = len(models)
    processes = mp.Queue()
    
    ctx = mp.get_context("spawn")
    # print(torch.multiprocessing.cpu_count())
    pool = ctx.Pool(num_processes)
    pool_list = []

    # GPU Warmup
    for _ in range(10):
        _ = models[0](dummy_inputs[0])

    for i in range(num_processes):
        res = pool.apply_async(test_exec_single, args=(models[i], dummy_inputs[0]))
    while not res.ready():
        pass
    
    print(2)

    start = time.time()
    res_list = []
    for i in range(num_processes):
        res = pool.apply(test_exec, args=(models[i], dummy_inputs))
        res_list.append(res)
    end = time.time()
    pool_list.append(end - start)

    start = time.time()
    res_list = []
    for i in range(num_processes):
        res = pool.apply_async(test_exec, args=(models[i], dummy_inputs))
        res_list.append(res)
    while not res_list[-1].ready():
        pass
    end = time.time()
    pool_list.append(end - start)

    pool.close()
    pool.join()

    print(pool_list)