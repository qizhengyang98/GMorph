#!/bin/bash
#SBATCH --job-name=tvm_scene_origin
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=gypsum-1080ti             # Partition to submit to (serial_requeue)
#SBATCH --mem=65536               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/work/qizhengyang_umass_edu/MTL_compiler/oslog/tvm_config/run_logs_%j.out            # File to which STDOUT will be written
#SBATCH --error=/work/qizhengyang_umass_edu/MTL_compiler/oslog/tvm_config/run_logs_%j.err            # File to which STDERR will be written
#SBATCH --gres=gpu:4
####efefSBATCH --cpus-per-task=6
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=qizhengyang@umass.edu

echo `pwd`
# echo "SLURM task ID: "$SLURM_ARRAY_TASK_ID
#module load cuda/11.3.1
set -x -e
##### Experiment settings #####
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/etc/profile.d/conda.sh" ]; then
        . "/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/etc/profile.d/conda.sh"
    else
        export PATH="/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<
conda init bash
conda activate tvm-build
sleep 1

cd /home/qizhengyang_umass_edu/MTL_compiler/test_metamorph/benchmark/
python ./tvm_compile.py \
--task=scene \
--model=origin \
--use_tune_config \
--enable_tune \

sleep 1
exit
