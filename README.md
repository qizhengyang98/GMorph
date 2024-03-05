# GMorph
AI-powered applications often involve multiple deep neural network (DNN)-based prediction tasks to support application level functionalities. However, executing multi-DNNs can be challenging due to the high resource demands and computation costs that increase linearly with the number of DNNs. Multi-task learning (MTL) addresses this problem by designing a multi-task model that shares parameters across tasks based on a single backbone DNN. This paper explores an alternative approach called model fusion: rather than training a single multi-task model from scratch as MTL does, model fusion fuses multiple task-specific DNNs which are pre-trained separately and can have heterogeneous architectures into a single multi-task model. We materialize model fusion in a software framework called GMorph to accelerate multi-DNN inference while maintaining task accuracy. GMorph features three technical contributions including graph mutation that fuses multi-DNNs into resource-efficient multi-task models, search-space sampling algorithms, and predictive filtering that reduce the high search costs. Our experiments show that GMorph can outperform MTL baselines and reduce the inference latency of multi-DNNs by 1.1-3× while meeting the target task accuracy.

# Description
## Structure
The folder "metamorph" contains all the source codes of GMorph. The folder "test_metamorph" contains the codes of all the benchmarks listed in the paper. The folder results is where all the log results will be stored. A "datasets" folder should be downloaded and placed under the root directory, which contains all the datasets used in the experiments.

## Dependencies
For hardware dependencies, CPU with 6 cores, 16G RAM, and a GPU (10G vRAM for benchmark-1,4,5, 20G for benchmark-2,3,6,7 are recommended).

For software dependencies, Linux OS, Python=3.8, and several Python packages listed in the *requirements.txt*.

The environment used in the paper: Ubuntu 20.04 LTS, NVIDIA Quadro RTX 8000 GPU.

## Environments and Set-up
Necessary dependencies are listed in the requirements.txt. A Conda environment is recommended for installation.

To set up the environments, firstly clone the repository. Go to the root folder and create a conda environment:
```
conda create -n gmorph python=3.8 ;
conda activate gmorph 
```
Then install all the necessary dependencies:
```
pip install -r requirements.txt ;
cd test_metamorph/transformers/ ;
pip install -e . 
```
Go to the root folder and install GMorph package:
```
cd ../.. ;
pip install metamorph/ 
```
Then all the necessary dependencies should be installed. To do a simple test, go to the folder metamorph/test and run
```
python test.py
```
If the computation graph of models is printed successfully, then installation is done.

## Run benchmarks
To run all the benchmarks and reproduce the results in the experiments, firstly download datasets and pre-trained single-task models from [Google Drive](https://drive.google.com/drive/folders/1Dtvd5eIDeDiseCAwCrj3_wrqjWsy3bq3?usp=sharing), put them under the corresponding folders:
- Put *datasets.zip* under the root directory and unzip it. There should be four folders under datasets: adience, ESOS, fer2013, VOCDetection;
- Put *salientNet.model*, *salientNet_vgg16.model*, *objectNet.model* under *test_metamorph/scene/pre_models*;
- Put *EmotionNet.model*, *EmotionNet_vgg13.model*, *ageNet.model*, *genderNet.model*, *genderNet_vgg11.model* under *test_metamorph/face/pre_models*;
- Put *age_gender.gz* under *metamorph/data*;
- Put *toy_vgg13.pt* under *metamorph/model*;
- Put *cola.zip*, *sst2.zip*, *multiclass.zip*, *salient.zip* under *test_metamorph/transformer_model* and unzip them.

Then we can Run GMorph for different benchmarks and generate well-trained multi-task models.

There are several shell scripts named *submit_xxx.sh* under the root directory, which are used to evaluate different benchmarks in this experiment. We will execute the shell scripts with proper arguments. The script *figure7table5.sh* is used to reproduce the results in Figure7, and the script *figure8.sh* is used to reproduce the results in Figure8 and Table5.

Under the *benchmark_scripts* folder, there are also separate scripts provided to run all the experiments for each benchmark without manually changing arguments, and which script corresponds to which experiment is written in the script *figure7table5.sh*.

There are some useful configurations on some arguments:
- policy_select: set *SimulatedAnnealing* when testing *GMorph*, set *LCBased* when testing *GMorph w P* and *GMorph w P+R*.
- log_name: the name of the log file, which saves useful intermediate information when GMorph is running.
- acc_drop_thres: the threshold of accuracy drop. 
- enable_filtering_rules: whether or not to enable rule-based filtering. Add this flag when testing *GMorph w P+R*, and remove this flag when testing *GMorph w P*. This flag is useful only when *policy_select=LCBased*.

Other arguments and flags do not need to be changed during evaluations. Note that the arguments of *batch_size* and *num_workers* can be smaller if GPU memory is not enough.

For different benchmark-n, run *submit_bn.sh*, and modify the flags or arguments as shown above. To run experiments without manually changing flags or arguments, go to the *benchmark_scripts* directory and run corresponding scripts. 

## Reproduce results
To reproduce the results shown in Figure7,8 and Table5, run scripts *figure7table5.sh* and *figure8.sh* accordingly. Note that running these scripts can be time-consuming, which basically runs all the experiments for all the benchmarks, so an alternative way is to run each experiment separately given the comments in the scripts.

When the shell script or commands inside are running, a corresponding log file will be generated under *results/log*. The log will record the architecture of the model, the accuracy and latency of the model, and the overall search time at the end of each iteration. Note that since GMorph is based on a random algorithm, the outcomes during the model searching and the final multi-task models may be similar but not exactly the same between different runs. It would be better to run each benchmark multiple times to generate multiple logs to minimize the influence of randomness.

To reproduce the results in Table4, run the shell script *table4.sh*, the latency of all-shared models and multi-task models found by TreeMTL in benchmark 1-4 will be printed. 

To reproduce the results in Table3, run the shell script *table3.sh*, the model architectures found by GMorph will be compiled by both PyTorch and TensorRT automatically, and the results, which are the latency of the models, will be printed. 12G GPU memory is needed for benchmark-6 and 15G is needed for benchmark-7.

# Citation
Welcome to cite our work if you find it helpful to your research
```
@article{gmorph2024,
  title={GMorph: Accelerating Multi-DNN Inference via Model Fusion},
  author={Yang, Qizheng and Yang, Tianyi and Xiang, Mingcan and Zhang, Lijun and Wang, Haoliang and Serafini, Marco and Guan, Hui},
  journal={Proceedings of Nineteenth European Conference on Computer Systems}
  year={2024}
}
```
