#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1

#SBATCH -o suff_comp_metrics_log.out

date '+[%H:%M:%S-%d/%m/%y]'

PROJECT_HOME=${PWD}

# Activate env
source ${PROJECT_HOME}/er_env/bin/activate

python ${PROJECT_HOME}/suff_comp_metrics_compute.py --path checkpoints/baseline/train/ --pred_filename pred_sst_dev_data --batch_size 512

date '+[%H:%M:%S-%d/%m/%y]'
