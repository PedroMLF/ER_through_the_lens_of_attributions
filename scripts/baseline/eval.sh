#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1

#SBATCH -o outputs/baseline/log_eval.out

date '+[%H:%M:%S-%d/%m/%y]'

PROJECT_HOME=${PWD}

# Activate env
source ${PROJECT_HOME}/er_env/bin/activate

MODEL_NAME=google/bigbird-roberta-base
BATCH_SIZE=32
SMALL_BATCH_SIZE=8
DATA_DIR=data/sa_data
CHECKPOINT_PATH=checkpoints/baseline/train

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

# Movies

echo "--- MOVIES ---"

python ${PROJECT_HOME}/src/approach.py \
    --model_name ${MODEL_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --data_dir ${DATA_DIR} \
    --test_data_filename "movies_dev-test.joblib" \
    --eval \
    --eval_batch_size ${SMALL_BATCH_SIZE} \
    --dataloader_num_workers 18 \
    --gpus 1

date '+[%H:%M:%S-%d/%m/%y]'

# Yelp

echo "--- YELP ---"

python ${PROJECT_HOME}/src/approach.py \
    --model_name ${MODEL_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --data_dir ${DATA_DIR} \
    --test_data_filename "yelp_test.joblib" \
    --eval \
    --eval_batch_size ${BATCH_SIZE} \
    --dataloader_num_workers 18 \
    --gpus 1

date '+[%H:%M:%S-%d/%m/%y]'

# Yelp-50

echo "--- YELP-50 ---"

python ${PROJECT_HOME}/src/approach.py \
    --model_name ${MODEL_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --data_dir ${DATA_DIR} \
    --test_data_filename "yelp-50.joblib" \
    --eval \
    --eval_batch_size ${BATCH_SIZE} \
    --dataloader_num_workers 18 \
    --gpus 1

date '+[%H:%M:%S-%d/%m/%y]'

# IMDB

echo "--- IMDB ---"

python ${PROJECT_HOME}/src/approach.py \
    --model_name ${MODEL_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --data_dir ${DATA_DIR} \
    --test_data_filename "imdb_test.joblib" \
    --eval \
    --eval_batch_size ${BATCH_SIZE} \
    --dataloader_num_workers 18 \
    --gpus 1

date '+[%H:%M:%S-%d/%m/%y]'

# Amazon

echo "--- AMAZON-MOVIES-TV ---"

python ${PROJECT_HOME}/src/approach.py \
    --model_name ${MODEL_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --data_dir ${DATA_DIR} \
    --test_data_filename "amazon_movies-tv.joblib" \
    --eval \
    --eval_batch_size ${BATCH_SIZE} \
    --dataloader_num_workers 18 \
    --gpus 1

date '+[%H:%M:%S-%d/%m/%y]'
