# ER through the Lens of Attributions

### 1. Setup

```
virtualenv -p /usr/bin/python3.9 er_env
source er_env/bin/activate
pip install pip==23.2.1
pip install -r requirements.txt
```

### 2. Populate data folder

```
./populate_data.sh
```

### 3. Run HParam search for Baseline and Joint experiments

```
sbatch scripts/baseline/hparam_search.sh
sbatch scripts/joint_attention/hparam_search.sh
sbatch scripts/joint_rollout/hparam_search.sh
```

### 4. Find Constrained approach bounds

- Use same baseline HParams.

```
sbatch scripts/constrained_attention/find_bound.sh
sbatch scripts/constrained_rollout/find_bound.sh
```

- This is a manual step:
    - Open Tensorboard for the `find_bound` runs.
    - Find the necessary values:
        - For `bound train init`, use half of the average of the initial `loss_annotation` value (rounded).
        - For `bound train minimum`, use the average minimum of `loss_annotation` (rounded).
        - For `bound validation`, use the average minimum of `val_avg_loss annotation` (rounded).

- Constrained Attention:
    - `loss_annotation` (initial):
        - [0.07477, 0.07784, 0.08364]
        - Mean: 0.07875
        - Bound Train Init: 0.035
    - `loss_annotation` (minimum):
        - [0.01929, 0.02287, 0.02437]
        - Mean: 0.02218
        - Bound Train Minimum: 0.023
    - `val_avg_loss annotation`:
        - [0.03037, 0.02989, 0.03027]
        - Mean: 0.03018
        - Bound Validation: 0.031

- Constrained Rollout:
    - `loss_annotation` (initial):
        - [0.06001, 0.06182, 0.0662]
        - Mean: 0.06268
        - Bound Train Init: 0.030
    - `loss_annotation` (minimum):
        - [0.02142, 0.0211, 0.02236]
        - Mean: 0.02163
        - Bound Train Minimum: 0.022
    - `val_avg_loss annotation`:
        - [0.03163, 0.03158, 0.03164]
        - Mean: 0.03162
        - Bound Validation: 0.031

### 5. Run HParam search for Constrained experiments

```
sbatch scripts/constrained_attention/hparam_search.sh
sbatch scripts/constrained_rollout/hparam_search.sh
```

### 6. Train models

```
sbatch scripts/baseline/train.sh
sbatch scripts/joint_attention/train.sh
sbatch scripts/joint_rollout/train.sh
sbatch scripts/constrained_attention/train.sh
sbatch scripts/constrained_rollout/train.sh
``` 

### 7. Run evaluation on all datasets

```
sbatch scripts/baseline/eval.sh
sbatch scripts/joint_attention/eval.sh
sbatch scripts/joint_rollout/eval.sh
sbatch scripts/constrained_attention/eval.sh
sbatch scripts/constrained_rollout/eval.sh
```

### 8. Get results

```
python gather_results.py outputs/baseline/log_eval.out test_f1_score --grouped_by_dataset
python gather_results.py outputs/joint_attention/log_eval.out test_f1_score --grouped_by_dataset
python gather_results.py outputs/joint_rollout/log_eval.out test_f1_score --grouped_by_dataset
python gather_results.py outputs/constrained_attention/log_eval.out test_f1_score --grouped_by_dataset
python gather_results.py outputs/constrained_rollout/log_eval.out test_f1_score --grouped_by_dataset
```

### 9. Get predictions and attributions

```
sbatch scripts/baseline/pred.sh
sbatch scripts/joint_attention/pred.sh
sbatch scripts/joint_rollout/pred.sh
sbatch scripts/constrained_attention/pred.sh
sbatch scripts/constrained_attention/pred_bound.sh
sbatch scripts/constrained_rollout/pred.sh
sbatch scripts/constrained_rollout/pred_bound.sh
```

### 10. Compute attribution plausibility scores

- This step outputs the values for Table 8.

```
python analysis/compute_plausibility_scores.py checkpoints/ all_analysis_data_pred_sst_dev_data.joblib --save
python analysis/compute_plausibility_scores.py checkpoints/ all_analysis_data_pred_yelp-50_data.joblib --save
python analysis/compute_plausibility_scores.py checkpoints/ all_analysis_data_pred_movies_dev-test_data.joblib --save
```

### 11. Run multiple lambdas experiment

```
sbatch scripts/multiple_lambdas/attention.sh
sbatch scripts/multiple_lambdas/rollout.sh
```

```
python gather_results.py outputs/multiple_lambdas/log_attention.out test_f1_score
python gather_results.py outputs/multiple_lambdas/log_rollout.out test_f1_score
python gather_losses.py outputs/multiple_lambdas/log_attention.out --dataset "SST-Dev"
python gather_losses.py outputs/multiple_lambdas/log_rollout.out --dataset "SST-Dev"
```

- Figure (`multiple_lambdas.pdf`):

```
python analysis/multiple_lambdas_plot.py
```

### 12. Generate figures and tables results

- Figure 2 (`results.pdf`):

```
python analysis/results.py
```

- Table 2 + Table 3:

```
python analysis/plausibility_scores.py checkpoints/plausibility_all_analysis_data_pred_sst_dev_data.joblib --metrics auc
```

And to get the loss values:

```
python gather_losses.py outputs/constrained_attention/log_find_bound.out --datasets=SST-Dev
python gather_losses.py outputs/joint_attention/log_eval.out --datasets=SST-Dev
python gather_losses.py outputs/constrained_attention/log_eval.out --datasets=SST-Dev
python gather_losses.py outputs/constrained_rollout/log_find_bound.out --datasets=SST-Dev
python gather_losses.py outputs/joint_rollout/log_eval.out --datasets=SST-Dev
python gather_losses.py outputs/constrained_rollout/log_eval.out --datasets=SST-Dev
```

- Table 4:

```
python analysis/plausibility_scores.py checkpoints/plausibility_all_analysis_data_pred_movies_dev-test_data.joblib --metrics auc --techniques attentions,rollout,IxG
python analysis/plausibility_scores.py checkpoints/plausibility_all_analysis_data_pred_yelp-50_data.joblib --metrics auc --techniques attentions,rollout,IxG
```

- Figure 4 + Figure 7:

```
python analysis/ood_perf_pred_figures.py
```

- Figure 6:

```
python analysis/compute_plausibility_scores_layer.py checkpoints/ all_analysis_data_pred_sst_dev_data.joblib --save
python analysis/plot_auc_per_layer.py
```
