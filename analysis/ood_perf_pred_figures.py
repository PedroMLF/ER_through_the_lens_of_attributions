import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dataset_to_str_map = {
    'SST-Test': 'SST-Test',
    'MOVIES': 'Movies',
    'IMDB': 'IMDB',
    'YELP': 'YELP',
    'AMAZON-MOVIES-TV': 'AMZ',
}

approach_to_str_map = {
    'BS': 'BS',
    'ER-A': 'ER + Att',
    'ER-R': 'ER + AttR',
    'ER-IxG': 'ER + IxG',
    'ER-C-A': 'ER-C + Att',
    'ER-C-R': 'ER-C + AttR',
    'ER-C-IxG': 'ER-C + IxG',
}

plot_kwargs = {'alpha': 0.8, 's': 70, 'edgecolor': 'k', 'linewidth': 1}


def get_plausibility_scores(results, metric, technique_to_keep, path):
    plausibility_scores_full = joblib.load(path)

    plausibility_score_to_keep = {}
    for approach, technique_values in plausibility_scores_full[metric].items():
        if approach not in results:
            continue
        plausibility_score_to_keep[approach] = {}
        for technique, values in technique_values.items():
            if technique != technique_to_keep:
                continue
            plausibility_score_to_keep[approach] = values

    for k, v in plausibility_score_to_keep.items():
        assert v

    return plausibility_score_to_keep


def prepare_df(results, plausibility_scores):
    df = pd.DataFrame({})
    for app, data in results.items():
        # Get in-domain predictor values
        id_values = data['SST-Dev']
        id_plausibility_scores = plausibility_scores[app]
        # Get the top and worst seeds based on plausibility metric
        for ds, ood_values in data.items():
            if ds in ['SST-Dev', 'MOVIES-128']:
                pass
            else:
                for ix, (id_value, ood_value, id_plaus) in enumerate(zip(id_values, ood_values, id_plausibility_scores)):
                    new_entry = {
                        'Approach': approach_to_str_map[app],
                        'Dataset': ds,
                        'OOD_Value': 100 * float(ood_value),
                        'ID_Value': 100 * float(id_value),
                        'ID_Plaus': 100 * float(id_plaus),
                    }

                    df = df.append(new_entry, ignore_index=True)
    
    return df


def plot_df(df, x, xlabel, ylabel, figsize, bbox_to_anchor, save_name, add_legend=True):
    sns.set(font_scale=1)
    sns.set_style("darkgrid")

    num_cols = len(df['Dataset'].unique())
    fig, ax = plt.subplots(1, num_cols, figsize=figsize, dpi=125, sharex=False, sharey=False)

    for ix, dataset in enumerate(df['Dataset'].unique()):
        if num_cols > 1:
            ax_to_plot = ax[ix]
        else:
            ax_to_plot = ax
        sns.scatterplot(
            df[(df['Dataset'] == dataset)],
            x=x,
            y='OOD_Value',
            style='Approach',
            hue='Approach',
            palette='colorblind',
            ax=ax_to_plot,
            **plot_kwargs,
        )

        ax_to_plot.set(xlabel=xlabel, ylabel=ylabel, title=dataset)

        if ix != 0:
            ax_to_plot.set_ylabel('')

        ax_to_plot.legend_.remove()

    if add_legend:
        plt.legend(title='Approach', bbox_to_anchor=bbox_to_anchor, ncols=len(df['Approach'].unique()), loc='lower center', frameon=False)
        for legend_handle in ax_to_plot.legend_.legendHandles:
            legend_handle.set_edgecolor('black')
            legend_handle.set_linewidth(0.6)

    plt.savefig(f"figures/{save_name}.pdf", dpi=500, bbox_inches='tight')


def prepare_df_from_ood(results, plausibility_scores):
    df = pd.DataFrame({})
    for app, data in results.items():
        # Get in-domain predictor values
        ood_plausibility_scores = plausibility_scores[app]
        # Get the top and worst seeds based on plausibility metric
        for ds, ood_values in data.items():
            if ds in ['SST-Test', 'SST-Dev', 'MOVIES-128']:
                pass
            else:
                for ix, (ood_value, ood_plaus) in enumerate(zip(ood_values, ood_plausibility_scores)):
                    new_entry = {
                        'Approach': approach_to_str_map[app],
                        'Dataset': ds,
                        'OOD_Value': 100 * float(ood_value),
                        'OOD_Plaus': 100 * float(ood_plaus),
                    }

                    df = df.append(new_entry, ignore_index=True)
    
    return df


def main():

    # Load results
    fn = "data_log_eval.joblib"
    results_paths = {
        'BS': f'outputs/baseline/{fn}',
        'ER-A': f'outputs/joint_attention/{fn}',
        'ER-R': f'outputs/joint_rollout/{fn}',
        'ER-IxG': f'outputs/joint_ixg_norm/{fn}',
        'ER-C-A': f'outputs/constrained_attention/{fn}',
        'ER-C-R': f'outputs/constrained_rollout/{fn}',
        'ER-C-IxG': f'outputs/constrained_ixg_norm/{fn}',
    }

    results = {k: joblib.load(v) for k, v in results_paths.items()}

    # Paths for the plausibility scores
    id_plausibility_scores_path = "checkpoints/plausibility_all_analysis_data_pred_sst_dev_data.joblib"
    ood_movies_plausibility_scores_path = "checkpoints/plausibility_all_analysis_data_pred_movies_dev-test_data.joblib"
    ood_yelp_plausibility_scores_path = "checkpoints/plausibility_all_analysis_data_pred_yelp-50_data.joblib"

    # In-Domain
    plausibility_scores = get_plausibility_scores(results=results, metric='auc', technique_to_keep='decompx_classifier', path=id_plausibility_scores_path)
    df = prepare_df(results, plausibility_scores=plausibility_scores)

    # Predictor: In-Domain Classification F1-Macro for SST-Test and Movies
    plot_df(
        df[(df['Dataset'].isin(['MOVIES'])) & (df['Approach'].isin(['BS', 'ER-C + Att', 'ER-C + AttR', 'ER-C + IxG']))],
        x='ID_Value',
        xlabel='ID F1-Macro',
        ylabel='OOD F1-Macro',
        # Use x = num_datasets * 4
        figsize=(4,4),
        bbox_to_anchor=(-0.1,-0.4),
        add_legend=False,
        save_name="sa-ood-prediction-cls",
    )

    # Predictor: In-Domain Plausibility with AUC + DecompX-Classifier for SST-Test and Movies
    plot_df(
        df[(df['Dataset'].isin(['MOVIES'])) & (df['Approach'].isin(['BS', 'ER-C + Att', 'ER-C + AttR', 'ER-C + IxG']))],
        x='ID_Plaus',
        xlabel='ID Plausibility AUC',
        ylabel=' ',
        # Use x = num_datasets * 4
        figsize=(4,4),
        bbox_to_anchor=(-0.15,-0.4),
        add_legend=False,
        save_name="sa-ood-prediction-auc",
    )

    # Predictor: In-Domain Classification F1-Macro for all datasets
    plot_df(
        df,#[~(df['Dataset'].isin(['YELP-50']))],
        x='ID_Value',
        xlabel='ID F1-Macro',
        ylabel='OOD F1-Macro',
        figsize=(24,4),
        bbox_to_anchor=(-1.8,-0.4),
        add_legend=False,
        save_name="sa-ood-prediction-full-cls",
    )

    # Predictor: In-Domain Plausibility with AUC + Attention for all datasets
    plausibility_scores = get_plausibility_scores(results=results, metric='auc', technique_to_keep='attentions', path=id_plausibility_scores_path)
    df = prepare_df(results, plausibility_scores=plausibility_scores)
    plot_df(
        df,
        x='ID_Plaus',
        xlabel='ID Plausibility AUC',
        ylabel='OOD F1-Macro',
        figsize=(24,4),
        bbox_to_anchor=(-1.8,-0.4),
        add_legend=False,
        save_name="sa-ood-prediction-full-auc-attn",
    )

    # Predictor: In-Domain Plausibility with AUC + Rollout for all datasets
    plausibility_scores = get_plausibility_scores(results=results, metric='auc', technique_to_keep='rollout', path=id_plausibility_scores_path)
    df = prepare_df(results, plausibility_scores=plausibility_scores)
    plot_df(
        df,
        x='ID_Plaus',
        xlabel='ID Plausibility AUC',
        ylabel='OOD F1-Macro',
        figsize=(24,4),
        bbox_to_anchor=(-1.8,-0.4),
        add_legend=False,
        save_name="sa-ood-prediction-full-auc-rollout",
    )

    # Predictor: In-Domain Plausibility with AUC + DecompX-Classifier for all datasets
    plausibility_scores = get_plausibility_scores(results=results, metric='auc', technique_to_keep='decompx_classifier', path=id_plausibility_scores_path)
    df = prepare_df(results, plausibility_scores=plausibility_scores)
    plot_df(
        df,
        x='ID_Plaus',
        xlabel='ID Plausibility AUC',
        ylabel='OOD F1-Macro',
        figsize=(24,4),
        bbox_to_anchor=(-1.8,-0.4),
        add_legend=False,
        save_name="sa-ood-prediction-full-auc-dxc",
    )

    # Predictor: In-Domain Plausibility with AP + DecompX-Classifier for all datasets
    plausibility_scores = get_plausibility_scores(results=results, metric='ap', technique_to_keep='decompx_classifier', path=id_plausibility_scores_path)
    df = prepare_df(results, plausibility_scores=plausibility_scores)
    plot_df(
        df,
        x='ID_Plaus',
        xlabel='ID Plausibility AP',
        ylabel='OOD F1-Macro',
        figsize=(24,4),
        bbox_to_anchor=(-1.8,-0.4),
        add_legend=False,
        save_name="sa-ood-prediction-full-ap-dxc",
    )

    # Predictor: In-Domain Plausibility with R@k + DecompX-Classifier for all datasets
    plausibility_scores = get_plausibility_scores(results=results, metric='recall', technique_to_keep='decompx_classifier', path=id_plausibility_scores_path)
    df = prepare_df(results, plausibility_scores=plausibility_scores)
    plot_df(
        df,
        x='ID_Plaus',
        xlabel='ID Plausibility R@k',
        ylabel='OOD F1-Macro',
        figsize=(24,4),
        bbox_to_anchor=(-2.5,-0.4),
        add_legend=True,
        save_name="sa-ood-prediction-full-rk-dxc",
    )

    # Finally, plot with OOD Plausibility predictors (using IxG), for OOD classification performance
    # - MOVIES attributions to MOVIES OOD
    # - YELP attributions to YELP OOD
    sns.set(font_scale=1)
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(1, 2, figsize=(8,4), dpi=125, sharex=False, sharey=False)

    # First plot MOVIES
    plausibility_scores = get_plausibility_scores(results=results, metric='auc', technique_to_keep='IxG', path=ood_movies_plausibility_scores_path)
    df = prepare_df_from_ood(results, plausibility_scores=plausibility_scores)

    ax_to_plot = ax[0]
    sns.scatterplot(
        df[(df['Dataset'] == 'MOVIES') & (df['Approach'].isin(['BS', 'ER-C + Att', 'ER-C + AttR', 'ER-C + IxG']))],
        x='OOD_Plaus',
        y='OOD_Value',
        style='Approach',
        hue='Approach',
        palette='colorblind',
        ax=ax_to_plot,
        **plot_kwargs,
    )

    ax_to_plot.set(xlabel='OOD Plausibility AUC', ylabel='OOD F1-Macro', title='MOVIES')
    ax_to_plot.legend_.remove()

    # Then YELP
    plausibility_scores = get_plausibility_scores(results=results, metric='auc', technique_to_keep='IxG', path=ood_yelp_plausibility_scores_path)
    df = prepare_df_from_ood(results, plausibility_scores=plausibility_scores)

    ax_to_plot = ax[1]
    sns.scatterplot(
        df[(df['Dataset'] == 'YELP-50') & (df['Approach'].isin(['BS', 'ER-C + Att', 'ER-C + AttR', 'ER-C + IxG']))],
        x='OOD_Plaus',
        y='OOD_Value',
        style='Approach',
        hue='Approach',
        palette='colorblind',
        ax=ax_to_plot,
        **plot_kwargs,
    )

    ax_to_plot.set(xlabel='OOD Plausibility AUC', ylabel='', title='YELP-50')
    ax_to_plot.legend_.remove()

    plt.legend(title='Approach', bbox_to_anchor=(-0.12,-0.4), ncols=len(df['Approach'].unique()), loc='lower center', frameon=False)
    for legend_handle in ax_to_plot.legend_.legendHandles:
        legend_handle.set_edgecolor('black')
        legend_handle.set_linewidth(0.6)

    plt.savefig("figures/sa-ood-prediction-ood-auc.pdf", dpi=500, bbox_inches='tight')


if __name__ == "__main__":
    if not os.path.exists("figures"):
        os.makedirs("figures")

    main()
