from typing import Dict, Callable

import timeit
import torch
import numpy as np
from torch.utils.data import DataLoader

from metamorph.graph.cmp_graph import ComputeGraph


import os, sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def test_latency(
    cmp_graph: ComputeGraph,
    dummy_input: torch.Tensor,
    timing_repeat=30, timing_number=30
) -> Dict[str, np.ndarray]:
    torch.manual_seed(0)
    cmp_graph.eval()
    with torch.no_grad():
        latency = (
            np.array(
                timeit.Timer(lambda: cmp_graph(dummy_input)).repeat(
                    repeat=timing_repeat, number=timing_number
                )
            )
            * 1000 / timing_number
        )
    latency = {"mean": np.mean(latency), "median": np.median(
        latency), "std": np.std(latency)}
    print("Inference Latency: %s \n" % (latency))
    return latency['mean']

def test_latency_cuda(
    cmp_graph: ComputeGraph,
    dummy_input: torch.Tensor,
    timing_repeat=30, timing_number=30
) -> Dict[str, np.ndarray]:
    cmp_graph.eval()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((timing_repeat, timing_number))
    input_size, input_device = dummy_input.size(), dummy_input.device
    sample_inputs = [torch.rand(input_size).to(input_device) for _ in range(100)]
    timing_mean, timing_std = None, None

    # GPU Warmup
    for _ in range(10):
         _ = cmp_graph(dummy_input)
    
    while timing_mean is None or timing_std > timing_mean/10:
        # Measure Performance
        with torch.no_grad():
            for i in range(timing_repeat):
                for j in range(timing_number):
                    torch.manual_seed(i+j)
                    start.record()
                    _ = cmp_graph(sample_inputs[(i+j)%len(sample_inputs)])
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

def test_accuracy(
    cmp_graph: ComputeGraph,
    f_test: Callable,
    test_loader: DataLoader,
    device: 'str'
):
    cmp_graph.eval()
    cmp_graph.to(device)
    return f_test(test_loader, cmp_graph, device)
