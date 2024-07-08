#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1

#SBATCH -o outputs/joint_attention/log_hparam.out

date '+[%H:%M:%S-%d/%m/%y]'

PROJECT_HOME=${PWD}

# Activate env
source ${PROJECT_HOME}/er_env/bin/activate

python ${PROJECT_HOME}/src/hparam_optuna.py \
    --save_dir "checkpoints/joint_attention/hparam_search" \
    --search_space_path "scripts/joint_attention/hparam_search_space.json" \
    --model_name "google/bigbird-roberta-base" \
    --data_dir "data/sa_data" \
    --train_data_filename "sst_train.joblib" \
    --dev_data_filename "sst_dev.joblib" \
    --test_data_filename "sst_test.joblib" \
    --dataloader_num_workers 18 \
    --seeds 0,1,2 \
    --metric_to_select "avg_loss_ce" \
    --metric_to_early_stop "avg_loss" \
    --attention_type "top_layer" \
    --attr_scaling 100 \
    --head_aggregation_method "mean" \

date '+[%H:%M:%S-%d/%m/%y]'
