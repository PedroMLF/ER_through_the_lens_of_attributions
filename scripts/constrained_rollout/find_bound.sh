#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1

#SBATCH -o outputs/constrained_rollout/log_find_bound.out

date '+[%H:%M:%S-%d/%m/%y]'

PROJECT_HOME=${PWD}

# Activate env
source ${PROJECT_HOME}/er_env/bin/activate

MODEL_NAME="google/bigbird-roberta-base"
BATCH_SIZE=32
DATA_DIR="data/sa_data"
OUTPUT_DIR="checkpoints/constrained_rollout"
EXPERIMENT_NAME="find_bound"

COUNT=0

for SEED in $(seq 0 2);
do
    python ${PROJECT_HOME}/src/approach.py \
    --model_name ${MODEL_NAME} \
    --data_dir ${DATA_DIR} \
    --train_data_filename "sst_train.joblib" \
    --dev_data_filename "sst_dev.joblib" \
    --test_data_filename "sst_test.joblib" \
    --train_batch_size ${BATCH_SIZE} \
    --eval_batch_size ${BATCH_SIZE} \
    --accumulate_grad_batches 1 \
    --learning_rate 3e-5 \
    --weight_decay 0 \
    --gradient_clip_val 0 \
    --max_epochs 25 \
    --scheduler "linear" \
    --early_stopping_patience 5 \
    --val_check_interval 1.0 \
    --metric_to_track "avg_loss" \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --dataloader_num_workers 18 \
    --gpus 1 \
    --attention_type "rollout_top_layer" \
    --attr_scaling 1 \
    --head_aggregation_method "mean" \
    --lambda_annotation_loss 0.0 \
    --random_seed ${SEED} \
    --log_every_n_steps 50 \
    --use_annotation_loss_only

    date '+[%H:%M:%S-%d/%m/%y]'

    CHECKPOINT_PATH=${OUTPUT_DIR}/${EXPERIMENT_NAME}/version_${COUNT}/*.ckpt

    echo "------------------------------------------------------------------------------------------------"
    echo "EVALUATING CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
    echo "------------------------------------------------------------------------------------------------"

    echo "COUNT: ${COUNT} || SEED: ${SEED}"

    # SST

    echo "--- SST-Dev ---"

    python ${PROJECT_HOME}/src/approach.py \
        --model_name ${MODEL_NAME} \
        --checkpoint_path ${CHECKPOINT_PATH} \
        --data_dir ${DATA_DIR} \
        --test_data_filename "sst_dev.joblib" \
        --eval \
        --eval_batch_size ${BATCH_SIZE} \
        --dataloader_num_workers 18 \
        --gpus 1

    date '+[%H:%M:%S-%d/%m/%y]'

    echo "--- SST-Test ---"

    python ${PROJECT_HOME}/src/approach.py \
        --model_name ${MODEL_NAME} \
        --checkpoint_path ${CHECKPOINT_PATH} \
        --data_dir ${DATA_DIR} \
        --test_data_filename "sst_test.joblib" \
        --eval \
        --eval_batch_size ${BATCH_SIZE} \
        --dataloader_num_workers 18 \
        --gpus 1

    date '+[%H:%M:%S-%d/%m/%y]'

    # Increase count for the model
    ((COUNT++))

done
