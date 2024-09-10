import os

import joblib
import numpy as np
import pandas as pd
import seaborn as sns

def print_averages(results, scale=False):
    scale_factor = 100 if scale else 1
    for k, v in results.items():
        print(f"{k}\n{'-'*10}")
        for kk, vv in v.items():
            print(f"{kk}: {scale_factor*np.mean(vv):.2f} ± {scale_factor*np.std(vv):.2f}")
        print()

def main():

    # Get paths to results
    print("Loading data...")
    fn = "data_log_eval.joblib"
    results_paths = {
        'Baseline': f'outputs/baseline/{fn}',
        'ER + Att': f'outputs/joint_attention/{fn}',
        'ER + AttR': f'outputs/joint_rollout/{fn}',
        'ER + IxG': f'outputs/joint_ixg_norm/{fn}',
        'ER-C + Att': f'outputs/constrained_attention/{fn}',
        'ER-C + AttR': f'outputs/constrained_rollout/{fn}',
        #'ER-C + IxG': f'outputs/constrained_ixg_norm/{fn}',
    }

    # Prepare dataframe
    results = {k: joblib.load(path) for k, path in results_paths.items()}

    print("Preparing data...")
    df = pd.DataFrame(
        [(app, ds, 100*float(value)) for app, data in results.items() for ds, values in data.items() for value in values],
        columns=['approach', 'dataset', 'value']
    )

    df = df[(df.dataset != "YELP-50")]

    # Plot
    sns.set(font_scale=1)
    sns.set_style("darkgrid")

    ax = sns.catplot(
        data=df,
        kind="box",
        col="dataset",
        y="approach",
        sharey=True,
        x="value",
        sharex=False,
        orient="h",

        width=0.5,

        palette='colorblind',

        legend=None,

        height=3,
        aspect=1.0,

        facet_kws=dict(gridspec_kws={"wspace":0.05}),
    )

    ax.map_dataframe(sns.stripplot, x="value", y="approach", alpha=0.4, color='black', dodge=True)

    ax.set(ylabel='Approach', xlabel='F1-Macro')
    ax.set_titles(template='{col_name}')

    ax.savefig("figures/results.pdf", dpi=500, bbox_inches='tight')

    print()
    print_averages(results, scale=True)

if __name__ == "__main__":
    if not os.path.exists("figures"):
        os.makedirs("figures")

    main()
