MODEL_NAME="google/bigbird-roberta-base"
BATCH_SIZE=32
DATA_DIR=data/sa_data
OUTPUT_DIR="DELETE/joint_ixg/"
EXPERIMENT_NAME="train"

SEED=0

python src/approach.py \
    --model_name checkpoints/joint_ixg_norm/train/version_0 \
    --data_dir ${DATA_DIR} \
    --train_data_filename "sst_train.joblib" \
    --dev_data_filename "sst_dev.joblib" \
    --test_data_filename "sst_test.joblib" \
    --train_batch_size ${BATCH_SIZE} \
    --eval_batch_size ${BATCH_SIZE} \
    --accumulate_grad_batches 1 \
    --weight_decay 0 \
    --gradient_clip_val 0 \
    --max_epochs 25 \
    --scheduler "linear" \
    --early_stopping_patience 10 \
    --val_check_interval 1.0 \
    --metric_to_track "avg_loss" \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --dataloader_num_workers 18 \
    --gpus 1 \
    --attention_type "IxG" \
    --attr_scaling 1 \
    --head_aggregation_method "mean" \
    --lambda_annotation_loss 0.0 \
    --random_seed ${SEED} \
    --log_every_n_steps 50 \
    --constrained_optimization \
    --constrained_optimization_bound_init 0.40 \
    --constrained_optimization_bound_min 0.25 \
    --constrained_optimization_validation_bound 0.35 \
    --constrained_optimization_smoothing 0.9 \
    --constrained_optimization_loss "guided" \
    --learning_rate 2e-5 \
    --constrained_optimization_lr 1e-1
