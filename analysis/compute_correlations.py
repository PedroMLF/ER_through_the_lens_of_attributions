import os
from argparse import ArgumentParser

import joblib
import numpy as np
import random
import torch
from scipy.stats import pearsonr, kendalltau

from tqdm import tqdm


def compute_correlations(data, approach_1, approach_2, technique_1, technique_2, corr_fn, fixed_version):
    random.seed(42)

    data_1 = data[approach_1][technique_1]
    data_2 = data[approach_2][technique_2]

    assert len(data_1) == len(data_2)
    assert len(data_1[0]) == len(data_2[0])

    versions_1 = list(data_1.keys())
    versions_2 = list(data_2.keys())
    num_examples = len(data_1[0])

    correlations = []

    for example_ix in range(num_examples):

        # Sample model versions
        if fixed_version:
            v1 = random.choice(versions_1)
            v2 = v1
        else:
            v1 = random.choice(versions_1)
            v2 = random.choice([v for v in versions_2 if v != v1])
        
        attrs_1 = data_1[v1][example_ix]
        attrs_2 = data_2[v2][example_ix]

        if len(attrs_1.shape) == 2:
            attrs_1 = attrs_1[-1]
        if len(attrs_2.shape) == 2:
            attrs_2 = attrs_2[-1]

        if technique_1 in ['IxG', 'decompx_classifier']:
            attrs_1 = torch.abs(attrs_1)
        if technique_2 in ['IxG', 'decompx_classifier']:
            attrs_2 = torch.abs(attrs_2)

        if corr_fn == "pearson":
            corr = pearsonr(attrs_1, attrs_2)[0]
        elif corr_fn == "kendall":
            corr = kendalltau(attrs_1, attrs_2)[0]
        correlations.append(corr)

    return correlations


def main(path, dataset_prefix, corr_fn):

    analysis_data_path = {
        'BS': f"checkpoints/baseline/train/{path}",
        'ER-A': f"checkpoints/joint_attention/train/{path}",
        'ER-R': f"checkpoints/joint_rollout/train/{path}",
        'ER-IxG': f"checkpoints/joint_ixg_norm/train/{path}",
        'ER-C-A': f"checkpoints/constrained_attention/train/{path}",
        'ER-C-R': f"checkpoints/constrained_rollout/train/{path}",
        'ER-C-IxG': f"checkpoints/constrained_ixg_norm/train/{path}",
    }

    data = {k:joblib.load(v) for k,v in tqdm(analysis_data_path.items())}

    # Define techniques
    if 'movies' in dataset_prefix:
        techniques = ['attentions', 'rollout', 'IxG']#, 'alti_aggregated', 'decompx', 'decompx_classifier']
    else:
        techniques = ['attentions', 'rollout', 'IxG', 'alti_aggregated', 'decompx', 'decompx_classifier']

    # Compute correlations
    all_corrs = {}

    # Compute correlation for fixed approaches
    # For a fixed approach, compute correlation between attribution techniques
    # An example output is the correlation between Baseline Attention and Baseline ALTI attributions
    # In this case, we use the same version to correlate attributions between examples
    print("\nComputing correlations for approaches...")
    all_corrs['approaches'] = {}
    for approach in data.keys():
        all_corrs['approaches'][approach] = {}
        print(f"\n-- {approach} --")
        for technique in techniques:
            all_corrs['approaches'][approach][technique] = {}
            print(technique)
            #aux_techniques = [at for at in techniques if at != technique]
            for aux_technique in techniques:
                print(f"\t{aux_technique}: ", end='')
                corrs = compute_correlations(
                    data,
                    approach_1=approach,
                    approach_2=approach,
                    technique_1=technique,
                    technique_2=aux_technique,
                    corr_fn=corr_fn,
                    # If we are comparing attributions for the same approach, I want to compare the same model version
                    fixed_version=False,
                )
                print(f"{np.mean(corrs):.2f} +- {np.std(corrs):.2f}")
                all_corrs['approaches'][approach][technique][aux_technique] = corrs


    # Compute correlation for fixed attribution techniques
    # For a fixed attribution technique, compute correlation between attributions across approaches
    # An example output is the correlation between Baseline Attention and ER+Att Attention
    # In this case, we use different version to correlation attributions between examples
    print("\nComputing correlation for attribution techniques...")
    all_corrs['techniques'] = {}
    for technique in techniques:
        all_corrs['techniques'][technique] = {}
        print(f"\n-- {technique} --")
        for approach in data.keys():
            all_corrs['techniques'][technique][approach] = {}
            print(approach)
            for aux_approach in data.keys():
                print(f"\t{aux_approach}: ", end='')
                corrs = compute_correlations(
                    data,
                    approach_1=approach,
                    approach_2=aux_approach,
                    technique_1=technique,
                    technique_2=technique,
                    corr_fn=corr_fn,
                    fixed_version=False,
                )
                print(f"{np.mean(corrs):.2f} +- {np.std(corrs):.2f}")
                all_corrs['techniques'][technique][approach][aux_approach] = corrs

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"correlations_{dataset_prefix}_{corr_fn}.joblib")
    print(f"Saving to: {save_path}")
    joblib.dump(all_corrs, save_path, compress=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("dataset_prefix", type=str)
    parser.add_argument("corr_fn", type=str)
    args = parser.parse_args()

    if args.corr_fn not in ["pearson", "kendall"]:
        raise Exception

    main(path=args.path, dataset_prefix=args.dataset_prefix, corr_fn=args.corr_fn)

    #python compute_correlations.py all_analysis_data_pred_sst-dev_data.joblib sst_dev kendall
