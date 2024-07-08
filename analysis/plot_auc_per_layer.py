import os

import joblib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


approach_to_str_map = {
    'BS': 'Baseline',
    'ER-A': 'ER+Att',
    'ER-R': 'ER+AttR',
    'ER-C-A': 'ER-C+Att',
    'ER-C-R': 'ER-C+AttR',
}

technique_to_str_map = {
    'decompx': 'DecompX',
    'alti_aggregated': 'ALTI',
    'attentions': 'Attention',
    'rollout': 'Attention-Rollout',
}

def main():
    x = joblib.load("checkpoints/plausibility_per_layer_all_analysis_data_pred_sst_dev_data.joblib")

    # Plot AUC w/ Attention
    fig, ax = plt.subplots(1,1, figsize=(4, 6), sharey=True)

    sns.set(font_scale=1)

    for ix, technique in enumerate(["attentions"]):

        # Get all results
        all_means = []
        for approach in x['auc'].keys():
            aux_means = []
            for layer_scores in x['auc'][approach][technique]:
                aux_means.append(np.mean(layer_scores))
            all_means.append(aux_means)

        #Plot
        ax_to_plot = ax
        sns.heatmap(np.array(all_means).T, annot=True, cmap='Blues', linewidths=0.1, ax=ax_to_plot)
        ax_to_plot.set_xticklabels([approach_to_str_map[app] for app in x['auc'].keys()])
        ax_to_plot.set_xticklabels(ax_to_plot.get_xticklabels(), rotation=30)
        plt.gca().invert_yaxis()
        if ix == 0:
            ax_to_plot.set_yticklabels(ax_to_plot.get_yticklabels(), rotation=0)
            ax_to_plot.set_ylabel("Layer")
        ax_to_plot.set_xlabel("Approach")
        ax_to_plot.set_title(f"AUC / {technique_to_str_map[technique]}")

    save_path = "figures/sa_auc_per_layer_att.png"
    print("Saving figure to: ", save_path)
    fig.savefig(save_path, dpi=500, bbox_inches='tight')

    # Plot AUC w/ ALTI and DecompX
    techniques = ["alti_aggregated", "decompx"]

    fig, ax = plt.subplots(1,len(techniques), figsize=(4*len(techniques), 6), sharey=True)

    sns.set(font_scale=1)

    for ix, technique in enumerate(techniques):

        # Get all results
        all_means = []
        for approach in x['auc'].keys():
            aux_means = []
            for layer_scores in x['auc'][approach][technique]:
                aux_means.append(np.mean(layer_scores))
            all_means.append(aux_means)

        #Plot
        ax_to_plot = ax[ix]
        sns.heatmap(np.array(all_means).T, annot=True, cmap='Blues', linewidths=0.1, ax=ax_to_plot)
        ax_to_plot.set_xticklabels([approach_to_str_map[app] for app in x['auc'].keys()])
        ax_to_plot.set_xticklabels(ax_to_plot.get_xticklabels(), rotation=30)
        plt.gca().invert_yaxis()
        if ix == 0:
            ax_to_plot.set_yticklabels(ax_to_plot.get_yticklabels(), rotation=0)
            ax_to_plot.set_ylabel("Layer")
        ax_to_plot.set_xlabel("Approach")
        ax_to_plot.set_title(f"AUC / {technique_to_str_map[technique]}")

    save_path = "figures/sa_auc_per_layer.png"
    print("Saving figure to: ", save_path)
    fig.savefig(save_path, dpi=500, bbox_inches='tight')


if __name__ == "__main__":
    if not os.path.exists("figures"):
        os.makedirs("figures")

    main()