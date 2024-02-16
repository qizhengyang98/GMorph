export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=~/tmp/nvidia-mps 
export CUDA_MPS_LOG_DIRECTORY=~/tmp/nvidia-log
nvidia-cuda-mps-control -d

# ps -ef | grep mps