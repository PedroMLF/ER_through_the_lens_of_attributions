#!/bin/bash
#SBATCH --time=03:30:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1

#SBATCH -o outputs/constrained_ixg_norm/log_hparam_2.out

date '+[%H:%M:%S-%d/%m/%y]'

PROJECT_HOME=${PWD}

# Activate env
source ${PROJECT_HOME}/er_env/bin/activate

python ${PROJECT_HOME}/src/hparam_optuna.py \
    --save_dir "checkpoints/constrained_ixg_norm/hparam_search" \
    --search_space_path "scripts/constrained_ixg/hparam_search_space.json" \
    --model_name "google/bigbird-roberta-base" \
    --data_dir "data/sa_data" \
    --train_data_filename "sst_train.joblib" \
    --dev_data_filename "sst_dev.joblib" \
    --test_data_filename "sst_test.joblib" \
    --dataloader_num_workers 18 \
    --seeds 0,1,2 \
    --metric_to_select "avg_loss_ce" \
    --metric_to_early_stop "avg_loss_ce" \
    --metric_to_track "avg_loss_ce" \
    --attention_type "IxG" \
    --attr_scaling 1 \
    --head_aggregation_method "mean" \
    --constrained_optimization \
    --constrained_optimization_bound_init 0.35 \
    --constrained_optimization_bound_min 0.24 \
    --constrained_optimization_validation_bound 0.35 \
    --constrained_optimization_smoothing 1.0 \
    --constrained_optimization_loss "guided"

date '+[%H:%M:%S-%d/%m/%y]'
