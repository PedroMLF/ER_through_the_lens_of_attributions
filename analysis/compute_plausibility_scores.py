import os
import random
from argparse import ArgumentParser

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from plausibility_metrics import compute_auc_score, compute_average_precision, compute_recall_at_k


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_results(base_path, fn):
    analysis_data_path = {
        'BS': f"{base_path}/baseline/train/{fn}",
        'ER-A': f"{base_path}/joint_attention/train/{fn}",
        'ER-R': f"{base_path}/joint_rollout/train/{fn}",
        'ER-IxG': f"{base_path}/joint_ixg_norm/train/{fn}",
        'ER-C-A': f"{base_path}/constrained_attention/train/{fn}",
        'ER-C-R': f"{base_path}/constrained_rollout/train/{fn}",
        'ER-C-IxG': f"{base_path}/constrained_ixg_norm/train/{fn}",
        'L-EXP_A': f"{base_path}/constrained_attention/find_bound/{fn}",
        'L-EXP_R': f"{base_path}/constrained_rollout/find_bound//{fn}",
        'L-EXP_IxG': f"{base_path}/constrained_ixg_norm/find_bound/{fn}",
    }

    results = {k:joblib.load(v) for k, v in tqdm(analysis_data_path.items())}

    return results


def gather_plausibibity_metric_results(results, techniques):

    # Set number of examples and versions
    num_examples = len(results["BS"]["predictions"][0]['preds'])
    print("Num Examples: ", num_examples)

    # Get annotation targets and tokens
    annotation_targets = results["BS"]["predictions"][0]["annotation_targets"]
    tokens = results["BS"]["predictions"][0]["tokens"]

    # Compute scores
    auc_scores = {}
    ap_scores = {}
    recall_scores = {}
    for approach in tqdm(results, desc="Approach: "):
        print(approach, end=" ... ")
        auc_scores[approach] = {}
        ap_scores[approach] = {}
        recall_scores[approach] = {}
        for technique in tqdm(techniques, desc="Technique: "):
            auc_scores[approach][technique] = []
            ap_scores[approach][technique] = []
            recall_scores[approach][technique] = []
            # For each model version, compute score for each example, and then get the mean
            for version in list(results[approach]["predictions"].keys()):
                aux_auc = []
                aux_ap = []
                aux_rec = []
                for example_ix in range(num_examples):
                    if not results[approach]["predictions"][0]["annotation_keep_loss"][example_ix]:
                        continue

                    labels = annotation_targets[example_ix]
                    scores = results[approach][technique][version][example_ix]

                    # If multiple layer keep only the top layer
                    if len(scores.shape) == 2:
                        scores = scores[-1]
                    
                    # If signed techniques, do absolute score
                    if technique in ['IxG', 'decompx_classifier']:
                        scores = torch.abs(scores)

                    aux_auc.append(compute_auc_score(labels, scores))
                    aux_ap.append(compute_average_precision(labels, scores))
                    aux_rec.append(compute_recall_at_k(labels, scores))

                auc_scores[approach][technique].append(np.mean(aux_auc))
                ap_scores[approach][technique].append(np.mean(aux_ap))
                recall_scores[approach][technique].append(np.mean(aux_rec))
    
    return {'auc': auc_scores, 'ap': ap_scores, 'recall': recall_scores}


def get_df_from_scores(scores, techniques, techniques_to_shortcut):

    df = pd.DataFrame({})

    for scores_str, scores_values in scores.items():
        for technique in techniques:
            for strategy_approach in scores_values:
                for v in scores_values[strategy_approach][technique]:
                    strategy = strategy_approach.split('_')[0]
                    approach = '-'.join(strategy_approach.split('_')[1:])
                    new_data = {
                        'Metric': scores_str,
                        'Approach': strategy_approach,
                        'Technique': technique,
                        'Strategy': approach,
                        'Score': 100*v
                    }

                    df = df.append(new_data, ignore_index=True)
                
    df['Technique'] = df['Technique'].map(techniques_to_shortcut)

    return df


def aggregate_and_print_results(df):

    means = df.groupby(['Metric', 'Technique', 'Approach']).mean()
    stds = df.groupby(['Metric', 'Technique', 'Approach']).std()

    strategies = [
        ('BS', '\\textbf{Baseline}'),
        ('ER-A', '\\textbf{ER + Att}'),
        ('ER-R', '\\textbf{ER + AttR}'), 
        ('ER-IxG', '\\textbf{ER + IxG}'), 
        ('ER-C-A', '\\textbf{ER-C + Att}'),
        ('ER-C-R', '\\textbf{ER-C + AttR}'),
        ('ER-C-IxG', '\\textbf{ER-C + IxG}'),
        ('L-EXP_A','$\\mathcal L_{\\text{expl}}$ (A)'),
        ('L-EXP_R','$\\mathcal L_{\\text{expl}}$ (R)'),
        ('L-EXP_IxG','$\\mathcal L_{\\text{expl}}$ (IxG)'),
    ]

    # Filter non existing strategies
    strategies = [s for s in strategies if s[0] in df['Approach'].unique().tolist()]

    for strategy, title in strategies:
        new_line = f"{title}"
        for metric in ['auc', 'ap', 'recall']:
            for technique in df['Technique'].unique().tolist():
                new_line += f" & {np.round(means.loc[metric, technique, strategy].item(), 1)} $\pm$ {np.round(stds.loc[metric, technique, strategy].item(), 1)}"
        new_line += " \\\\"
        print(new_line)


def main(base_path, fn, save):

    print(f"{'-' * 10} {fn} {'-' * 10}")

    seed_everything(0)

    # Define techniques to use
    techniques = ['attentions', 'rollout', 'IxG', 'alti_aggregated', 'decompx_classifier', 'decompx']
    techniques_to_shortcut = {
        'attentions': 'Att',
        'rollout': 'AttR',
        'IxG': 'IxG',
        'alti': 'AL',
        'alti_aggregated': 'ALTI',
        'decompx_classifier': 'DX-C',
        'decompx': 'DX',
        'attributions': 'Attr',
    }

    # Load results
    results = load_results(base_path, fn)

    # Filter non-existing techniques
    techniques = [t for t in techniques if t in results['BS']]

    # Get all plausibility metric scores
    all_plausibility_metrics_scores = gather_plausibibity_metric_results(results, techniques)
    df_plausibility_metrics = get_df_from_scores(all_plausibility_metrics_scores, techniques, techniques_to_shortcut)
    aggregate_and_print_results(df_plausibility_metrics)

    if save:
        save_path = os.path.join(base_path, "plausibility_" + fn)
        print("Saving data to: ", save_path)
        joblib.dump(all_plausibility_metrics_scores, save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("base_path", type=str)
    parser.add_argument("fn", type=str)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    main(base_path=args.base_path, fn=args.fn, save=args.save)
