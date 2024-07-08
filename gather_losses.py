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

def main(path, datasets):

    datasets = datasets.split(",")

    # Save results to txt file
    results_txt_path = os.path.dirname(path) + os.path.sep + "results_losses.txt"
    fp = open(results_txt_path, 'w')

    results = {}

    key = None
    for line in open(path):
        line = line.strip()

        # Update dataset being read
        pattern_line = check_pattern(line)
        if pattern_line:
            key = pattern_line
            if key not in results:
                results[key] = {'test_avg_loss_ce': [], 'test_avg_loss_annotation': []}

        # Store values
        if "test_avg_loss_ce" in line and key:
            results[key]["test_avg_loss_ce"].append(float(line.split("test_avg_loss_ce")[-1].strip()))
            # Reset key, since loss_ce appears after loss_annotation
            #key = None

        if "test_avg_loss_annotation" in line and key:
            results[key]["test_avg_loss_annotation"].append(float(line.split("test_avg_loss_annotation")[-1].strip()))

    # Filter datasets
    results = {k:v for k, v in results.items() if k in datasets}

    fp.write("RESULTS:\n")
    for key, values in results.items():
        fp.write(key + "\n")
        for subkey, subvalues in values.items():
            if not subvalues: continue
            fp.write(f"{subkey}: {[np.round(v,5) for v in subvalues]}\n")
        fp.write("\n")
    fp.write(f"{'--'*15}\n")

    fp.write("RESULTS PER DATASET:\n")
    for key, values in results.items():
        fp.write(key + "\n")
        for subkey, subvalues in values.items():
            if not subvalues: continue
            result_mean = np.round(np.mean(subvalues), 3)
            result_std = np.round(np.std(subvalues), 3)
            fp.write(f"{subkey} - Mean / STD : {str(result_mean).replace('.', ',')} \t {str(result_std).replace('.', ',')}\n")
        fp.write("\n")

    # Save data to joblib
    fp.write("\n")
    save_path = os.path.join(
        os.path.split(path)[0],
        f"data_{os.path.splitext(os.path.split(path)[-1])[0]}_losses.joblib",
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
    parser.add_argument("--datasets", type=str, default="SST-Dev")
    args = parser.parse_args()

    main(args.path, args.datasets)
