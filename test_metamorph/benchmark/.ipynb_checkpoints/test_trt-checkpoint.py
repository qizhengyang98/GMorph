import torch
import torch.nn as nn
import tensorrt
import torch_tensorrt

import numpy as np

import argparse
import timeit

from benchmark_toy import toy_origin as b1_origin
from benchmark_toy import toy_best_LC_norule as b1_t002

from benchmark_face16 import face16_origin as b2_origin
from benchmark_face16 import face16_SA_002 as b2_t002

from benchmark_face import face_origin as b3_origin
from benchmark_face import face_best_LC_norule_t002 as b3_t002

from benchmark_scene import scene_origin as b4_origin
from benchmark_scene import scene_SA_t002 as b4_t002

from benchmark_scene_2 import scene_origin as b5_origin
from benchmark_scene_2 import scene_SA002 as b5_t002

from benchmark_vit import vit_origin as b6_origin
from benchmark_vit import vit_SA_t002 as b6_t002

from benchmark_glue import glue_origin as b7_origin
from benchmark_glue import glue_SA_t002 as b7_t002


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

    print("Inference Latency: %s" % (latency))
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
    print("Inference Latency: %s \n" % (latency))
    return latency['mean']

if __name__ == "__main__":
    torch.manual_seed(0)

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # sample_input = torch.randn(1,3,224,224).to(DEVICE)
    shape = [1,1,48,48]
    sample_input = torch.randn(shape).to(DEVICE)
    
    # benchmark-1
    model = b1_origin().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str = f"Benchmark-1 origin: PyTorch {latency}, TRT {latency_trt} \n"
    
    model = b1_t002().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-1 GMorph: PyTorch {latency}, TRT {latency_trt} \n\n"
    print(print_str)
    
    
    shape = [1,3,224,224]
    sample_input = torch.randn(shape).to(DEVICE)
    
    # benchmark-2
    model = b2_origin().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-2 origin: PyTorch {latency}, TRT {latency_trt} \n"
    
    model = b2_t002().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-2 GMorph: PyTorch {latency}, TRT {latency_trt} \n\n"
    print(print_str)
    
    # benchmark-3
    model = b3_origin().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-3 origin: PyTorch {latency}, TRT {latency_trt} \n"
    
    model = b3_t002().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-3 GMorph: PyTorch {latency}, TRT {latency_trt} \n\n"
    print(print_str)
    
    # benchmark-4
    model = b4_origin().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-4 origin: PyTorch {latency}, TRT {latency_trt} \n"
    
    model = b4_t002().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-4 GMorph: PyTorch {latency}, TRT {latency_trt} \n\n"
    print(print_str)
    
    # benchmark-5
    model = b5_origin().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-5 origin: PyTorch {latency}, TRT {latency_trt} \n"
    
    model = b5_t002().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-5 GMorph: PyTorch {latency}, TRT {latency_trt} \n\n"
    print(print_str)
    
    # benchmark-6
    model = b6_origin().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-6 origin: PyTorch {latency}, TRT {latency_trt} \n"
    
    model = b6_t002().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-6 GMorph: PyTorch {latency}, TRT {latency_trt} \n\n"
    print(print_str)
    
    
    shape = [1,128]
    sample_input = torch.ones((1,128), dtype=torch.int).to(DEVICE)
    
    # benchmark-7
    model = b7_origin().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.int32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-7 origin: PyTorch {latency}, TRT {latency_trt} \n"
    
    model = b7_t002().eval().to(DEVICE)
    traced_model = torch.jit.trace(model, [sample_input])
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=shape, dtype=torch.int32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)
    latency = test_trt_latency(model, sample_input)
    latency_trt = test_trt_latency(trt_model, sample_input)
    print_str += f"Benchmark-7 GMorph: PyTorch {latency}, TRT {latency_trt}"
    print(print_str)
    