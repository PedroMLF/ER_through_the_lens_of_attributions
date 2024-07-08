from argparse import ArgumentParser

import joblib
import numpy as np

tech2str = {
    'attentions': 'Att',
    'rollout': 'AttR',
    'IxG': 'IxG',
    'alti_aggregated': 'ALTI',
    'decompx_classifier': 'DX-C',
    'decompx': 'DX',
}

approach2str = {
    'BS': 'Baseline',
    'ER-A': 'ER + Att',
    'ER-R': 'ER + AttR',
    'ER-C-A': 'ER-C + Att',
    'ER-C-R': 'ER-C + AttR',
    'L-EXP_A': 'L_expl (Att)',
    'L-EXP_R': 'L_expl (AttR)',
}

def main(data_path, metrics, techniques, print_std):

    print("Loading data from: ", data_path)
    data = joblib.load(data_path)

    metrics = metrics.split(",")
    techniques = techniques.split(",")

    for metric, metric_data in data.items():
        if metric not in metrics:
            continue
        print("\n\nMETRIC: ", metric)
        print("TECHNIQUES: ", ', '.join([tech2str[t] for t in techniques]))
        for approach, approach_data in metric_data.items():
            str_to_print = approach2str[approach]
            for technique_ix, (technique, technique_data) in enumerate(approach_data.items()):
                if technique not in techniques:
                    continue
                tch_data = [t * 100 for t in technique_data]
                str_to_print += f" & {np.mean(tch_data):.1f}"
                if print_std:
                    str_to_print += f" \pm {np.std(tch_data):.1f}"
            print(str_to_print)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--metrics", type=str, default="auc,ap,recall")
    parser.add_argument("--techniques", type=str, default="attentions,rollout,IxG,alti_aggregated,decompx_classifier,decompx")
    parser.add_argument("--print_std", action="store_true")
    args = parser.parse_args()

    main(data_path=args.data_path, metrics=args.metrics, techniques=args.techniques, print_std=args.print_std)
