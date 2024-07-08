#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1

#SBATCH -o outputs/constrained_rollout/log_pred_bound.out

date '+[%H:%M:%S-%d/%m/%y]'

PROJECT_HOME=${PWD}

# Activate env
source ${PROJECT_HOME}/er_env/bin/activate

EXPERIMENT_PATH="checkpoints/constrained_rollout/find_bound"

MODEL_NAME="google/bigbird-roberta-base"
DATA_DIR="data/sa_data"

BATCH_SIZE=32
SMALL_BATCH_SIZE=8

# SST-DEV / YELP-50

for DATA_PATH in "sst_dev" "yelp-50"
do
    echo -e "\n\n\n --- DATASET: ${DATA_PATH} --- \n"
    PRED_PATH_FILENAME="pred_${DATA_PATH}_data.joblib"

    for VERSION in $(seq 0 2); do

        CHECKPOINT_PATH=${EXPERIMENT_PATH}/version_${VERSION}/*.ckpt

        echo "-----------------------------------"
        echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
        echo "-----------------------------------"

        python ${PROJECT_HOME}/src/approach.py \
            --model_name ${MODEL_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --data_dir ${DATA_DIR} \
            --predict_data_filename "${DATA_PATH}.joblib" \
            --predict \
            --eval_batch_size ${BATCH_SIZE} \
            --dataloader_num_workers 18 \
            --enable_progress_bar \
            --gpus 1

        # Run captum metrics
        echo -e "\n --- RUNNING CAPTUM --- \n"

        echo -e "IxG:\n"

        python ${PROJECT_HOME}/compute_captum_attributions.py \
            --path ${EXPERIMENT_PATH}/version_${VERSION}/ \
            --pred_path_filename ${PRED_PATH_FILENAME} \
            --batch_size ${BATCH_SIZE} \
            --technique "IxG"

        # Run extra attribution metrics
        echo -e "\n --- RUNNING EXTRA ATTRIBUTIONS --- \n"

        echo -e "Decompx:\n"

        python ${PROJECT_HOME}/compute_extra_attributions.py \
            --path ${EXPERIMENT_PATH}/version_${VERSION}/ \
            --pred_path_filename ${PRED_PATH_FILENAME} \
            --batch_size ${SMALL_BATCH_SIZE} \
            --do_decompx

        echo -e "\nRollout:\n"

        python ${PROJECT_HOME}/compute_extra_attributions.py \
            --path ${EXPERIMENT_PATH}/version_${VERSION}/ \
            --pred_path_filename ${PRED_PATH_FILENAME} \
            --batch_size ${BATCH_SIZE} \
            --do_attention_rollout

        echo -e "\nALTI:\n"

        python ${PROJECT_HOME}/compute_extra_attributions.py \
            --path ${EXPERIMENT_PATH}/version_${VERSION}/ \
            --pred_path_filename ${PRED_PATH_FILENAME} \
            --batch_size ${BATCH_SIZE} \
            --do_alti

    done

    date '+[%H:%M:%S-%d/%m/%y]'

    python ${PROJECT_HOME}/prepare_pred_data.py \
        ${EXPERIMENT_PATH} \
        pred_${DATA_PATH}_data \
        IxG,decompx,decompx_classifier,alti,alti_aggregated \
        rollout

    date '+[%H:%M:%S-%d/%m/%y]'
done

# MOVIES

DATA_PATH="movies_dev-test"
echo -e "\n\n\n --- DATASET: ${DATA_PATH} --- \n"
PRED_PATH_FILENAME="pred_${DATA_PATH}_data.joblib"

for VERSION in $(seq 0 2); do

    CHECKPOINT_PATH=${EXPERIMENT_PATH}/version_${VERSION}/*.ckpt

    echo "-----------------------------------"
    echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
    echo "-----------------------------------"

    python ${PROJECT_HOME}/src/approach.py \
        --model_name ${MODEL_NAME} \
        --checkpoint_path ${CHECKPOINT_PATH} \
        --data_dir ${DATA_DIR} \
        --predict_data_filename "${DATA_PATH}.joblib" \
        --predict \
        --eval_batch_size ${SMALL_BATCH_SIZE} \
        --dataloader_num_workers 18 \
        --enable_progress_bar \
        --gpus 1

    # Run captum metrics
    echo -e "\n --- RUNNING CAPTUM --- \n"

    echo -e "IxG:\n"

    python ${PROJECT_HOME}/compute_captum_attributions.py \
        --path ${EXPERIMENT_PATH}/version_${VERSION}/ \
        --pred_path_filename ${PRED_PATH_FILENAME} \
        --batch_size ${SMALL_BATCH_SIZE} \
        --technique "IxG"

    # Run extra attribution metrics
    echo -e "\nRollout:\n"

    python ${PROJECT_HOME}/compute_extra_attributions.py \
        --path ${EXPERIMENT_PATH}/version_${VERSION}/ \
        --pred_path_filename ${PRED_PATH_FILENAME} \
        --batch_size ${SMALL_BATCH_SIZE} \
        --do_attention_rollout

done

date '+[%H:%M:%S-%d/%m/%y]'

python ${PROJECT_HOME}/prepare_pred_data.py \
    ${EXPERIMENT_PATH} \
    pred_${DATA_PATH}_data \
    IxG \
    rollout

date '+[%H:%M:%S-%d/%m/%y]'
