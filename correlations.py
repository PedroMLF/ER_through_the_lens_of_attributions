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

    versions = list(range(len(data_1)))
    num_examples = len(data_1[0])

    correlations = []

    for example_ix in range(num_examples):

        # Sample model versions
        if fixed_version:
            v1 = random.choice(versions)
            v2 = v1
        else:
            v1, v2 = random.sample(versions, k=2)
        
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

    correlations.append(np.mean(correlations))

    return correlations


def main(corr_fn):

    fn = "all_analysis_data_pred_sst_dev_data.joblib"

    analysis_data_path = {
        'BS': f"checkpoints/baseline/train/{fn}",
        'ER-A': f"checkpoints/joint_attention/train/{fn}",
        'ER-R': f"checkpoints/joint_rollout/train/{fn}",
        'ER-C-A': f"checkpoints/constrained_attention/train/{fn}",
        'ER-C-R': f"checkpoints/constrained_rollout/train/{fn}",
    }

    data = {k:joblib.load(v) for k,v in tqdm(analysis_data_path.items())}

    # Define techniques
    techniques = ['attentions', 'rollout', 'IxG', 'alti_aggregated', 'decompx', 'decompx_classifier']

    # Compute correlations
    all_corrs = {}

    # Compute correlation across approaches
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
                    fixed_version=True,
                )
                print(f"{np.mean(corrs):.2f} +- {np.std(corrs):.2f}")
                all_corrs['approaches'][approach][technique][aux_technique] = corrs

    # Compute correlation across attribution techniques
    all_corrs['techniques'] = {}
    for technique in techniques:
        all_corrs['techniques'][technique] = {}
        print(f"\n-- {technique} --")
        for approach in data.keys():
            all_corrs['techniques'][technique][approach] = {}
            print(approach)
            #aux_approaches = [aa for aa in data.keys() if aa != approach]
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

    joblib.dump(all_corrs, f"correlations-{corr_fn}_teste.joblib", compress=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("corr_fn", type=str)
    args = parser.parse_args()

    if args.corr_fn not in ["pearson", "kendall"]:
        raise Exception

    main(args.corr_fn)
