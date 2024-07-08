import os
import re
from argparse import ArgumentParser

import joblib
import numpy as np

def check_pattern(text):
    pattern = r"---\s+(.*?)\s+---\n?" 
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def main(path, metric, grouped_by_dataset):

    # Save results to txt file
    results_txt_path = os.path.dirname(path) + os.path.sep + "results.txt"
    fp = open(results_txt_path, 'w')

    fp.write(f"Processing: {path}\n")

    results = {}

    key = None
    for line in open(path):
        line = line.strip()

        # Update dataset being read
        pattern_line = check_pattern(line)
        if pattern_line:
            key = pattern_line
            if key not in results:
                results[key] = []

        # Store values
        if metric in line and key:
            if ":" in line:
                results[key].append(float(line.split(metric + "\':")[-1].strip(",").strip()))
            else:
                results[key].append(float(line.split(metric)[-1].strip()))
            # If each dataset is evaluated for all ckpts before the next, keep the key
            if not grouped_by_dataset:
                key = None

    fp.write("RESULTS:\n")
    for key, values in results.items():
        fp.write(f"{key}: {[np.round(100*v,3) for v in values]}\n")
    fp.write(f"{'--'*15}\n")

    full_str_to_print = ""

    fp.write(f"RESULTS PER DATASET ({metric}):\n")
    for key, values in results.items():
        result_mean = 100*np.mean(values)
        result_std = 100*np.std(values)
        fp.write(f"{key} - Mean: {result_mean:.5f} - STD: {result_std:.5f}\n")

        full_str_to_print += f"{str(result_mean).replace('.',',')}\t"
        full_str_to_print += f"{str(result_std).replace('.',',')}\t"

    fp.write(f"\n{full_str_to_print}\n")

    # Save data to joblib
    fp.write("\n")
    save_path = os.path.join(
        os.path.split(path)[0],
        f"data_{os.path.splitext(os.path.split(path)[-1])[0]}.joblib",
    )
    fp.write(f"Saving data to: {save_path}\n")
    joblib.dump(results, save_path)

    fp.close()

    # Print results
    for line in open(results_txt_path):
        print(line.strip('\n'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("metric", type=str)
    parser.add_argument(
        "--grouped_by_dataset",
        action="store_true",
        help="Use if results are grouped by dataset and not by checkpoint"
    )
    args = parser.parse_args()

    main(args.path, args.metric, args.grouped_by_dataset)
