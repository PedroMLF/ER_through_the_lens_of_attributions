import os

import joblib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Comment out those that should be left out of the plot
approach_to_str_map = {
    'BS': 'Baseline',
    'ER-A': 'ER+Att',
#    'ER-R': 'ER+AttR',
#    'ER-IxG': 'ER+IxG',
    'ER-C-A': 'ER-C+Att',
    'ER-C-R': 'ER-C+AttR',
    'ER-C-IxG': 'ER-C+IxG',
}

technique_to_str_map = {
    'decompx': 'DecompX',
    'alti_aggregated': 'ALTI',
    'attentions': 'Attention',
    'rollout': 'Attention-Rollout',
}

def main():
    x = joblib.load("checkpoints/plausibility_per_layer_all_analysis_data_pred_sst_dev_data.joblib")
    
    """
    # Plot AUC w/ All ID
    techniques = ["attentions", "rollout", "alti_aggregated", "decompx"]

    fig, ax = plt.subplots(1,len(techniques), figsize=(5*len(techniques), 6), sharey=True)

    sns.set(font_scale=1)

    for ix, technique in enumerate(techniques):

        # Get all results
        all_means = []
        for approach in approach_to_str_map.keys():
            aux_means = []
            for layer_scores in x['auc'][approach][technique]:
                aux_means.append(100 * np.mean(layer_scores))
            all_means.append(aux_means)

        #Plot
        ax_to_plot = ax[ix]
        sns.heatmap(np.array(all_means).T, annot=True, cmap='Blues', linewidths=0.1, cbar=False, ax=ax_to_plot)
        ax_to_plot.set_xticklabels([approach_to_str_map[app] for app in approach_to_str_map.keys()])
        ax_to_plot.set_xticklabels(ax_to_plot.get_xticklabels(), rotation=90)
        plt.gca().invert_yaxis()
        if ix == 0:
            ax_to_plot.set_yticklabels(ax_to_plot.get_yticklabels(), rotation=0)
            ax_to_plot.set_ylabel("Layer")
        ax_to_plot.set_xlabel("Approach")
        ax_to_plot.set_title(f"AUC / {technique_to_str_map[technique]}")

    save_path = "figures/sa_auc_per_layer_all_id.pdf"
    print("Saving figure to: ", save_path)
    fig.savefig(save_path, dpi=500, bbox_inches='tight')
    """

    # Plot AUC w/ Attention and DecompX, smaller version
    techniques = ["attentions", "decompx"]

    fig, ax = plt.subplots(1,len(techniques), figsize=(4*len(techniques), 4), sharey=False)

    sns.set(font_scale=1)

    for ix, technique in enumerate(techniques):

        layers_to_keep = [0, 4, 8, 10, 11]

        # Get all results
        all_means = []
        for approach in approach_to_str_map.keys():
            aux_means = []
            for layer_ix, layer_scores in enumerate(x['auc'][approach][technique]):
                if layer_ix in layers_to_keep:
                    aux_means.append(100 * np.mean(layer_scores))
            all_means.append(aux_means)

        #Plot
        ax_to_plot = ax[ix]
        sns.heatmap(np.array(all_means).T, annot=True, cmap='Blues', linewidths=0.1, cbar=False, ax=ax_to_plot)
        ax_to_plot.set_xticklabels([approach_to_str_map[app] for app in approach_to_str_map.keys()])
        ax_to_plot.set_xticklabels(ax_to_plot.get_xticklabels(), rotation=90)
        ax_to_plot.invert_yaxis()

        if ix == 0:
            # Fix ticklabels
            ax_to_plot.set_yticklabels(layers_to_keep, rotation=0)
            ax_to_plot.set_ylabel("Layer")
        else:
            ax_to_plot.set_yticklabels('')
        ax_to_plot.set_xlabel("Approach")
        ax_to_plot.set_title(f"AUC / {technique_to_str_map[technique]}")

    save_path = "figures/sa_auc_per_layer_attention_dx_smaller.pdf"
    print("Saving figure to: ", save_path)
    fig.savefig(save_path, dpi=500, bbox_inches='tight')

    """
    # Plot OOD
    x = joblib.load("checkpoints/plausibility_per_layer_all_analysis_data_pred_yelp-50_data.joblib")

    # Plot AUC w/ All OOD
    techniques = ["attentions", "rollout", "alti_aggregated", "decompx"]

    fig, ax = plt.subplots(1,len(techniques), figsize=(5*len(techniques), 6), sharey=True)

    sns.set(font_scale=1)

    for ix, technique in enumerate(techniques):

        # Get all results
        all_means = []
        for approach in approach_to_str_map.keys():
            aux_means = []
            for layer_scores in x['auc'][approach][technique]:
                aux_means.append(100 * np.mean(layer_scores))
            all_means.append(aux_means)

        #Plot
        ax_to_plot = ax[ix]
        sns.heatmap(np.array(all_means).T, annot=True, cmap='Blues', linewidths=0.1, cbar=False, ax=ax_to_plot)
        ax_to_plot.set_xticklabels([approach_to_str_map[app] for app in approach_to_str_map.keys()])
        ax_to_plot.set_xticklabels(ax_to_plot.get_xticklabels(), rotation=90)
        plt.gca().invert_yaxis()
        if ix == 0:
            ax_to_plot.set_yticklabels(ax_to_plot.get_yticklabels(), rotation=0)
            ax_to_plot.set_ylabel("Layer")
        ax_to_plot.set_xlabel("Approach")
        ax_to_plot.set_title(f"AUC / {technique_to_str_map[technique]}")

    save_path = "figures/sa_auc_per_layer_all_ood_yelp.pdf"
    print("Saving figure to: ", save_path)
    fig.savefig(save_path, dpi=500, bbox_inches='tight')


    # Plot AUC w/ All OOD (Movies)
    x = joblib.load("checkpoints/plausibility_per_layer_all_analysis_data_pred_movies_dev-test_data.joblib")

    techniques = ["attentions", "rollout"]

    fig, ax = plt.subplots(1,len(techniques), figsize=(5*len(techniques), 6), sharey=True)

    sns.set(font_scale=1)

    for ix, technique in enumerate(techniques):

        # Get all results
        all_means = []
        for approach in approach_to_str_map.keys():
            aux_means = []
            for layer_scores in x['auc'][approach][technique]:
                aux_means.append(100 * np.mean(layer_scores))
            all_means.append(aux_means)

        #Plot
        ax_to_plot = ax[ix]
        sns.heatmap(np.array(all_means).T, annot=True, cmap='Blues', linewidths=0.1, cbar=False, ax=ax_to_plot)
        ax_to_plot.set_xticklabels([approach_to_str_map[app] for app in approach_to_str_map.keys()])
        ax_to_plot.set_xticklabels(ax_to_plot.get_xticklabels(), rotation=90)
        plt.gca().invert_yaxis()
        if ix == 0:
            ax_to_plot.set_yticklabels(ax_to_plot.get_yticklabels(), rotation=0)
            ax_to_plot.set_ylabel("Layer")
        ax_to_plot.set_xlabel("Approach")
        ax_to_plot.set_title(f"AUC / {technique_to_str_map[technique]}")

    save_path = "figures/sa_auc_per_layer_all_ood_movies.pdf"
    print("Saving figure to: ", save_path)
    fig.savefig(save_path, dpi=500, bbox_inches='tight')
    """

if __name__ == "__main__":
    if not os.path.exists("figures"):
        os.makedirs("figures")

    main()