import os 
import numpy as np
import tvm
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime


target = tvm.target.cuda()
log_file = "mtl_tvm.log"
dtype = "float32"

input_name = "input0"

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 2000,
    "early_stopping": 600,
    "measure_option": autotvm.measure_option(
        builder = autotvm.LocalBuilder(timeout=10),
        runner = autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}


def get_network(model, shape_list):
    input_name = shape_list[0][0]
    mod, params = relay.frontend.from_pytorch(model, shape_list)
    return mod, params #, shape_list[0][0]


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="mtltvm_tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))
        
        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)
        
        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
                
        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial = n_trial,
            early_stopping = early_stopping,
            measure_option = measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )
        
    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
    
    
def tune_and_evaluate(mod, params, input_shape, tuning_opt=tuning_option, log_name=log_file):
    # set log file
    tuning_opt["log_filename"] = log_name
    # extract workloads from relay program
    print("Extract tasks ... ")
#     mod, params, input_shape = get_network(model, shape_list)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
    
    # run tuning tasks
    print("Tuning ... ")
    tune_tasks(tasks, **tuning_opt)
    
    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compiling ... ")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
            
        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input(input_name, data_tvm)
        
        # evaluate 
        print("Evaluate inference time cost ... ")
        print(module.benchmark(dev, number=1, repeat=600))