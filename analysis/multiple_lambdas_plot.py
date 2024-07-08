import os

import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
import seaborn as sns


def build_df(full_results_path, full_losses_path, constrained_results_path, constrained_losses_path, lambdas, num_seeds, datasets):

    # Load full results
    full_results = joblib.load(full_results_path)

    results = {ld:{} for ld in lambdas}
    for i in range(len(lambdas)):
        for dataset, scores in full_results.items():
            assert len(scores) == num_seeds * len(lambdas)
            results[lambdas[i]][dataset] = scores[num_seeds * i : num_seeds * (i+1)]

    # Load full losses
    full_losses = joblib.load(full_losses_path)

    losses = {ld:{} for ld in lambdas}
    for i in range(len(lambdas)):
        for dataset, loss_values in full_losses.items():
            for aux_loss_values in loss_values.values():
                if aux_loss_values: assert len(aux_loss_values) == num_seeds * len(lambdas), print(aux_loss_values)
            losses[lambdas[i]][dataset] = {aux_loss: aux_loss_values[num_seeds * i : num_seeds * (i+1)] for aux_loss, aux_loss_values in loss_values.items()}

    # Load constrained results / losses:
    if constrained_results_path and constrained_losses_path:
        constrained_results = joblib.load(constrained_results_path)
        constrained_losses = joblib.load(constrained_losses_path)

        constrained_results_to_plot = {k: v[:num_seeds] for k, v in constrained_results.items()}
        constrained_losses_to_plot = {k: {kk: vv[:num_seeds] for kk, vv in v.items()} for k, v in constrained_losses.items()}

    # Build dataframe
    df = pd.DataFrame(columns=['lambda', 'seed', 'dataset', 'score', 'loss_ce', 'loss_annot'])

    for ld in lambdas:
        # Get scores
        dataset_scores = results[ld]
        data_to_skip = []
        for dataset in datasets:
            scores = dataset_scores[dataset]
            loss_values = losses[ld][dataset]
            for seed, score in enumerate(scores):
                new_entry = {
                    'lambda': ld,
                    'seed': seed,
                    'dataset': dataset,
                    'score': score,
                }

                # Get losses
                new_entry['loss_ce'] = loss_values['test_avg_loss_ce'][seed]
                new_entry['loss_annot'] = loss_values['test_avg_loss_annotation'][seed]

                if new_entry['loss_ce'] > 99 and 'dev' in dataset:
                    print(f"Skipping entry for lambda: {ld} -- seed: {seed} -- dataset: {dataset} -- ce_loss: {new_entry['loss_ce']}")
                    data_to_skip.append((ld, seed))

                elif (new_entry['lambda'], new_entry['seed']) in data_to_skip:
                    print(f"\tSkipping entry for lambda: {ld} -- seed: {seed} -- dataset: {dataset} -- ce_loss: {new_entry['loss_ce']}")

                elif ld in [2, 1000]:
                    pass
                
                else:
                    df = df.append(new_entry, ignore_index=True)

    # Add constrained results to dataframe
    if constrained_results_path and constrained_losses_path:
        for dataset in datasets:
            scores = constrained_results_to_plot[dataset]
            loss_values = constrained_losses_to_plot[dataset]
            for seed, score in enumerate(scores):
                new_entry = {
                    'lambda': 'constrained',
                    'seed': seed,
                    'dataset': dataset,
                    'score': score,
                }

                # Get losses
                new_entry['loss_ce'] = loss_values['test_avg_loss_ce'][seed]
                new_entry['loss_annot'] = loss_values['test_avg_loss_annotation'][seed]

                df = df.append(new_entry, ignore_index=True)

    return df


def plot_df_summarized(dfs, annotation_loss_bounds, dataset, title):

    # Set plot style
    sns.reset_orig()
    sns.set_context('paper')
    sns.set_style('whitegrid')
    sns.set_palette("colorblind")
    sns.set(rc={"lines.linewidth": 1, "font.size": 120})

    markers = ['o', 's', '^', 'v', 'p', '*', 'P']
    markers_edge_widths = [0.4, 1.0]
    colorblind_palette = sns.color_palette("colorblind")
    colors = [colorblind_palette[0], colorblind_palette[1]]
    ecolors = [colorblind_palette[0], colorblind_palette[1]]
    linestyles = ['-', '--']

    assert len(colors) >= len(dfs)

    fig, ax = plt.subplots(dpi=150)

    for df_ix, df in enumerate(dfs):

        # Subset data
        aux_df = df[(df.dataset == dataset)].copy()

        # Get data
        gb_aux_df = aux_df.groupby(['lambda'], as_index=False).agg({'loss_ce': ['mean', 'std'], 'loss_annot': ['mean', 'std']})
        lambdas = gb_aux_df['lambda'].tolist()

        xs = gb_aux_df['loss_annot']['mean'].tolist()
        ys = gb_aux_df['loss_ce']['mean'].tolist()
        xerrs = gb_aux_df['loss_annot']['std'].tolist()
        yerrs = gb_aux_df['loss_ce']['std'].tolist()

        # Plot
        for i in range(len(xs)):
            ax.errorbar(
                xs[i],
                ys[i],
                xerr=xerrs[i],
                yerr=yerrs[i],
                marker=markers[i],
                markeredgecolor='k',
                markeredgewidth=markers_edge_widths[df_ix],
                markersize=10,
                color=colors[df_ix],
                ecolor=colors[df_ix],
                elinewidth=1.5,
                label=f'λ = {lambdas[i]}',
                alpha=0.8,
            )

        # Annotation Loss Bound
        if 'Dev' in dataset:
            annotation_loss_mean = np.mean(annotation_loss_bounds[df_ix])
            annotation_loss_std = np.std(annotation_loss_bounds[df_ix])
            ax.axvline(annotation_loss_mean, color=colors[df_ix], linestyle=linestyles[df_ix], linewidth=1.5, alpha=0.8)
            #ax.fill_betweenx([-1, 1], annotation_loss_mean - annotation_loss_std, annotation_loss_mean + annotation_loss_std, alpha=0.2, color='blue')

        ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    # Create separate handles and labels for color and markers
    color_handles = [plt.Line2D([0], [0], color=color, marker='o', markersize=10, markeredgecolor='k', markeredgewidth=markers_edge_widths[i], linestyle=linestyles[i], linewidth=1) for i, color in enumerate(colors)]
    color_labels = ['ER + Att', 'ER + AttR']
    marker_handles = [plt.Line2D([0], [0], color='k', marker=marker, markersize=10, linestyle='None') for marker in markers]
    marker_labels = [f'λ = {lambdas[i]}' for i in range(len(xs))]

    # Manually change the constrained entry
    for i, ml in enumerate(marker_labels):
        if "constrained" in ml:
            marker_labels[i] = "Constrained"

    # Add legend with custom arrangement
    ax.legend(handles=color_handles + marker_handles, labels=color_labels + marker_labels, loc='upper right', ncol=2)

    #plt.legend(loc="upper right", ncol=2)

    # Set axis labels
    ax.set_title(title)
    
    ax.set_xlabel("Explanation Loss")
    ax.set_ylabel("CE Loss")

    save_path = "figures/multiple_lambdas.pdf"
    print("Saving figure to: ", save_path)
    fig.savefig(save_path, dpi=500, bbox_inches='tight')

    plt.show()


def main():
    # Manually set the experimental choices, to split the results
    print("Preparing data ...")
    num_seeds = 5
    lambdas = [0, 0.5, 1.0, 5.0, 10, 100]

    df_attention = build_df(
        full_results_path = "outputs/multiple_lambdas/data_log_attention.joblib",
        full_losses_path = "outputs/multiple_lambdas/data_log_attention_losses.joblib",
        constrained_results_path = "outputs/constrained_attention/data_log_eval.joblib",
        constrained_losses_path = "outputs/constrained_attention/data_log_eval_losses.joblib",
        lambdas=lambdas,
        num_seeds=num_seeds,
        datasets=['SST-Dev'],
    )

    df_rollout = build_df(
        full_results_path = "outputs/multiple_lambdas/data_log_rollout.joblib",
        full_losses_path = "outputs/multiple_lambdas/data_log_rollout_losses.joblib",
        constrained_results_path = "outputs/constrained_rollout/data_log_eval.joblib",
        constrained_losses_path = "outputs/constrained_rollout/data_log_eval_losses.joblib",
        lambdas=lambdas,
        num_seeds=num_seeds,
        datasets=['SST-Dev'],
    )

    annotation_loss_bounds = {'ER-C-A': [0.031], 'ER-C-R': [0.031]}

    # Plot
    print("Plotting ...")
    plot_df_summarized(
        [df_attention, df_rollout],
        [annotation_loss_bounds['ER-C-A'], annotation_loss_bounds['ER-C-R']],
        dataset='SST-Dev',
        title="SST-Dev",
    )

if __name__ == "__main__":
    if not os.path.exists("figures"):
        os.makedirs("figures")

    main()