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
        'ER-C-A': f"{base_path}/constrained_attention/train/{fn}",
        'ER-C-R': f"{base_path}/constrained_rollout/train/{fn}",
        #'L-EXP_A': f"{base_path}/constrained_attention/find_bound/{fn}",
        #'L-EXP_R': f"{base_path}/constrained_rollout/find_bound//{fn}",
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
    for approach in tqdm(results, desc="Approach: "):
        print(approach, end=" ... ")
        auc_scores[approach] = {}
        for technique in tqdm(techniques, desc="Technique: "):
            auc_scores[approach][technique] = [ [] for _ in range(12) ]

            for version in list(results[approach]["predictions"].keys()):
                aux_auc = [[] for _ in range(12)]
                for example_ix in range(num_examples):
                    if not results[approach]["predictions"][0]["annotation_keep_loss"][example_ix]:
                        continue

                    labels = annotation_targets[example_ix]
                    scores = results[approach][technique][version][example_ix]

                    for layer_ix, score in enumerate(scores):
                        aux_auc[layer_ix].append(compute_auc_score(labels, score))

                for metric_scores, aux_metric_scores in [(auc_scores, aux_auc)]:
                    for layer_ix, scores in enumerate(aux_metric_scores):
                        metric_scores[approach][technique][layer_ix].append(np.mean(scores))

    return {'auc': auc_scores}


def main(base_path, fn, save):

    print(f"{'-' * 10} {fn} {'-' * 10}")

    seed_everything(0)

    # Define techniques to use
    techniques = ['attentions', 'rollout', 'alti_aggregated', 'decompx']
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

    if save:
        save_path = os.path.join(base_path, "plausibility_per_layer_" + fn)
        print("Saving data to: ", save_path)
        joblib.dump(all_plausibility_metrics_scores, save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("base_path", type=str)
    parser.add_argument("fn", type=str)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    main(base_path=args.base_path, fn=args.fn, save=args.save)
