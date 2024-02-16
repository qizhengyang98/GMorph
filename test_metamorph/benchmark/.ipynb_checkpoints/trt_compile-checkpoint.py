import torch
import torch.nn as nn
import tensorrt
import torch_tensorrt

import numpy as np

import argparse
import timeit

# from benchmark import results_toymodel, results_scene, results_face16, results_face
# from benchmark import results_toymodel_t0, results_scene_t0, results_face16_t0, results_face_t0
# from benchmark import results_toymodel_t002, results_scene_t002, results_face16_t002, results_face_t002
# from benchmark_scene_2 import scene_origin, scene_SA000, scene_SA001, scene_LC001, scene_SA002, scene_LCR002
# from benchmark_glue import glue_origin, glue_SA_t002
# from benchmark_vit import vit_origin, vit_SA_t002
# from benchmark_scene import scene_origin, scene_SA_t002, scene_best_LC_norule_t002, scene_best_LC_rule_t002
# from benchmark_scene_2 import scene_origin as scene_origin_2
from benchmark_toy import toy_origin, toy_half_conv, toy_all_conv, toy_best_LC_norule, toy_best_LC_rule, toy_best_LC_rule_t002, toy_SA_t0, toy_best_LC_norule_t0, toy_best_LC_rule_t0

# from benchmark_face import face_origin


# parser = argparse.ArgumentParser()
# parser.add_argument('--task', help="test task: ['toy', 'scene', 'face16', 'face']", type=str)
# parser.add_argument('--model', help="test model: ['origin', 'merge_half_conv', 'merge_all_conv', 'SA', 'LC_wo_rule', 'LC_w_rule']", type=str)
# parser.add_argument('--thres', help="acc drop threshold:[0,0.01,0.02]", type=float)

# args = parser.parse_args()

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
    sample_input = torch.randn(1,1,48,48).to(DEVICE)
    # sample_input = torch.ones((1,128), dtype=torch.int).to(DEVICE)
    # model = scene_origin().eval().to(DEVICE)
    # model2 = scene_origin_2().eval().to(DEVICE)
    # model = scene_origin().eval().to(DEVICE)
    # model = scene_SA_t002().eval().to(DEVICE)
    # model3 = scene_best_LC_norule_t002().eval().to(DEVICE)
    # model4 = scene_best_LC_rule_t002().eval().to(DEVICE)
    # model = toy_origin().eval().to(DEVICE)
    model = toy_best_LC_norule().eval().to(DEVICE)
    # model3 = toy_best_LC_norule_t0().eval().to(DEVICE)
    # model4 = toy_best_LC_rule_t0().eval().to(DEVICE)
    # model5 = toy_all_conv().eval().to(DEVICE)
    
    
    # model = scene_origin().eval().to(DEVICE)
#     model = scene_SA000().eval().to(DEVICE)
#     model = scene_SA001().eval().to(DEVICE)
#     model = scene_LC001().eval().to(DEVICE)
    # model = scene_SA002().eval().to(DEVICE)
#     model = scene_LCR002().eval().to(DEVICE)
    # model = glue_origin().eval().to(DEVICE)
    # model = glue_SA_t002().eval().to(DEVICE)
    # model = vit_origin().eval().to(DEVICE)
    # model = vit_SA_t002().eval().to(DEVICE)
    
    
#     if args.thres == 0:
#         from benchmark import results_toymodel_t0, results_scene_t0, results_face16_t0, results_face_t0
#         res_toymodel = results_toymodel_t0
#         res_scene = results_scene_t0
#         res_face16 = results_face16_t0
#         res_face = results_face_t0
#     elif args.thres == 0.01:
#         from benchmark import results_toymodel, results_scene, results_face16, results_face
#         res_toymodel = results_toymodel
#         res_scene = results_scene
#         res_face16 = results_face16
#         res_face = results_face
#     elif args.thres == 0.02:
#         from benchmark import results_toymodel_t002, results_scene_t002, results_face16_t002, results_face_t002
#         res_toymodel = results_toymodel_t002
#         res_scene = results_scene_t002
#         res_face16 = results_face16_t002
#         res_face = results_face_t002
#     else:
#         print('Invalid thres: must be either 0 or 0.01 or 0.02')
    
#     if args.task == 'toy':
#         model = res_toymodel[args.model].to(DEVICE)
#         sample_input = torch.rand(1,1,48,48).to(DEVICE)
#     elif args.task == 'scene':
#         model = res_scene[args.model].to(DEVICE)
#     elif args.task == 'face16':
#         model = res_face16[args.model].to(DEVICE)
#     elif args.task == 'face':
#         model = res_face[args.model].to(DEVICE)
#     else:
#         print('Invalid task: task must be either toy or scene or face16 or face')

    # model = face_origin().eval().to(DEVICE)
        
    # trt_module = torch_tensorrt.compile(
    #     model,
    #     inputs=[
    #         torch_tensorrt.Input(sample_input.shape),
    #     ],
    #     min_block_size=1
    # )
    # print(trt_module)
    
#     test_trt_latency(model1, sample_input)
#     test_trt_latency(model2, sample_input)
#     test_trt_latency(model3, sample_input)
#     test_trt_latency(model4, sample_input)
#     # test_trt_latency(model5, sample_input)
#     print('-------------------------------------------------------')
    
#     test_trt_latency_cuda(model1, sample_input)
#     test_trt_latency_cuda(model2, sample_input)
#     test_trt_latency_cuda(model3, sample_input)
#     test_trt_latency_cuda(model4, sample_input)
#     # test_trt_latency_cuda(model5, sample_input)
#     print('-------------------------------------------------------')
    
    traced_model = torch.jit.trace(model, [sample_input])
    
    # trt_model = torch_tensorrt.compile(traced_model,
    #                               inputs=[torch_tensorrt.Input(shape=[1,128], dtype=torch.int32),],
    #                               enabled_precisions={torch.float32},
    #                               truncate_long_and_double=True)
    trt_model = torch_tensorrt.compile(traced_model,
                                  inputs=[torch_tensorrt.Input(shape=[1,1,48,48], dtype=torch.float32),],
                                  enabled_precisions={torch.float32},
                                  truncate_long_and_double=True)

    test_trt_latency(trt_model, sample_input)
    test_trt_latency(trt_model, sample_input)
    test_trt_latency_cuda(trt_model, sample_input)
    