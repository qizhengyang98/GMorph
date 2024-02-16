import torch
import torch.nn as nn

import numpy as np

import argparse
import timeit

from benchmark_toy import toy_all_conv as b1_allshared
from benchmark_mtl import b1_mtl

from benchmark_face16 import face16_all_conv as b2_allshared
from benchmark_mtl import b2_mtl

from benchmark_face import face_all_conv as b3_allshared
from benchmark_mtl import b3_mtl

from benchmark_scene import scene_all_conv as b4_allshared
from benchmark_mtl import b4_mtl


def test_trt_latency(trt_model, input_x, timing_number=30, timing_repeat=30):
    # GPU Warmup
    for _ in range(10):
        _ = trt_model(input_x)

    latency = (
            np.array(timeit.Timer(lambda:trt_model(input_x)).repeat(repeat=timing_repeat, number=timing_number))
            * 1000
            / timing_number
        )
    latency = {"mean": np.mean(latency), "median": np.median(latency), "std":np.std(latency)}

    # print("Inference Latency: %s" % (latency))
    return latency['mean']

def test_trt_latency_cuda(trt_model, input_x, timing_repeat=30, timing_number=30):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((timing_repeat, timing_number))
    timing_mean, timing_std = None, None

    # GPU Warmup
    for _ in range(10):
        _ = trt_model(input_x)
    
    while timing_mean is None or timing_std > timing_mean/10:
        # Measure Performance
        with torch.no_grad():
            for i in range(timing_repeat):
                for j in range(timing_number):
                    torch.manual_seed(i+j)
                    start.record()
                    _ = trt_model(input_x)
                    end.record()
                    # Wait for GPU synchronization
                    torch.cuda.synchronize()
                    timings[i][j] = start.elapsed_time(end)
        
        timing_mean = np.mean(np.mean(timings, axis=1))
        timing_median = np.median(timings)
        timing_std = np.mean(np.std(timings, axis=1, ddof=1))
    
    latency = {"mean": timing_mean, "median": timing_median, "std": timing_std}
    # print("Inference Latency: %s \n" % (latency))
    return latency['mean']

if __name__ == "__main__":
    torch.manual_seed(0)

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    shape = [1,1,48,48]
    sample_input = torch.randn(shape).to(DEVICE)
    
    # benchmark-1
    model = b1_allshared().eval().to(DEVICE)
    latency = test_trt_latency(model, sample_input)
    model = b1_mtl().eval().to(DEVICE)
    latency_mtl = test_trt_latency(model, sample_input)
    print_str = f"Benchmark-1: AllShared {latency}, TreeMTL {latency_mtl} \n"
    print(print_str)
    
    
    shape = [1,3,224,224]
    sample_input = torch.randn(shape).to(DEVICE)
    
    # benchmark-2
    model = b2_allshared().eval().to(DEVICE)
    latency = test_trt_latency(model, sample_input)
    model = b2_mtl().eval().to(DEVICE)
    latency_mtl = test_trt_latency(model, sample_input)
    print_str += f"Benchmark-2: AllShared {latency}, TreeMTL {latency_mtl} \n"
    print(print_str)

    # benchmark-3
    model = b3_allshared().eval().to(DEVICE)
    latency = test_trt_latency(model, sample_input)
    model = b3_mtl().eval().to(DEVICE)
    latency_mtl = test_trt_latency(model, sample_input)
    print_str += f"Benchmark-3: AllShared {latency}, TreeMTL {latency_mtl} \n"
    print(print_str)
    
    # benchmark-4
    model = b4_allshared().eval().to(DEVICE)
    latency = test_trt_latency(model, sample_input)
    model = b4_mtl().eval().to(DEVICE)
    latency_mtl = test_trt_latency(model, sample_input)
    print_str += f"Benchmark-4: AllShared {latency}, TreeMTL {latency_mtl} \n"
    print(print_str)