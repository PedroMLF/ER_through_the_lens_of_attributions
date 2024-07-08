import argparse
import os
from glob import glob

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from pytorch_lightning import seed_everything


def load_attention_extra_data(data_path, filename, attentions_to_load):
    """Returns dictionary like {version: {approach: [attention_values_0 ... attention_values_n]}}"""
    tmp_data = {}

    for attention in attentions_to_load:
        tmp_data[attention] = {}
        paths = glob(os.path.join(data_path, "version_*", f"attention_{attention}_{filename}.joblib"))
        if not len(paths):
            raise Exception(f"Can't find attention flow files")

        for path in sorted(paths, key=lambda x: int(x.split("version_")[-1].split("/")[0])):
            print(f"Loading path: {path}")
            tmp_data[attention][int(path.split("version_")[-1].split("/")[0])] = joblib.load(path)

    # Assert all loaded dicts have the same keys
    #all_keys = [list(tmp_data[t].keys()) for t in attentions_to_load]
    #assert all(keys == all_keys[0] for keys in all_keys)

    return tmp_data

def load_attributions_data(data_path, filename, techniques_to_load):
    """Returns dictionaries like {technique: {version: {'attributions': [example_0_attributions, ..., example_n_attributions]}}}"""
    tmp_data = {}

    for technique in techniques_to_load:
        tmp_data[technique] = {}
        technique_paths = glob(os.path.join(data_path, "version_*", f"attributions_{filename}_{technique}.joblib"))
        if not len(technique_paths):
            raise Exception(f"Can't find files for {technique}")
        
        for path in sorted(technique_paths, key=lambda x: int(x.split("version_")[-1].split("/")[0])):
            print(f"Loading path: {path}")
            data_to_append = joblib.load(path)
            if "attributions" in data_to_append:
                data_to_append = data_to_append["attributions"]
            data_to_append = [x.float() if x.dtype == torch.bfloat16 else x for x in data_to_append]

            tmp_data[technique][int(path.split("version_")[-1].split("/")[0])] = data_to_append

    # Assert all loaded dicts have the same keys
    all_keys = [list(tmp_data[t].keys()) for t in techniques_to_load]
    assert all(keys == all_keys[0] for keys in all_keys)

    return tmp_data

def load_pred_data(data_path, filename):
    """Returns dictionaries like {version: {key: value}}"""
    tmp_data = {}

    paths = glob(os.path.join(data_path, "version_*", f"{filename}.joblib"))
    if not len(paths):
        raise Exception(f"Can't find prediction files")

    for path in sorted(paths, key=lambda x: int(x.split("version_")[-1].split("/")[0])):
        print(f"Loading path: {path}")
        tmp_data[int(path.split("version_")[-1].split("/")[0])] = joblib.load(path)

    # Assert all loaded dicts have the same keys
    all_keys = [list(tmp_data[v].keys()) for v in tmp_data]
    assert all(keys == all_keys[0] for keys in all_keys)

    return tmp_data

def main(args):

    # Get versions
    existing_versions = sorted([int(x.split('version_')[-1]) for x in glob(os.path.join(args.data_path, "version_*"))])

    # Parse techniques and attentions to load
    techniques_to_load = args.techniques_to_load.split(",")
    attentions_to_load = args.attentions_to_load.split(",")

    # Load attributions data
    print("\nLoading attributions data ...")
    attributions_data = load_attributions_data(
        data_path=args.data_path,
        filename=args.filename,
        techniques_to_load=techniques_to_load
    )

    # Load pred data
    print("\nLoading pred data ...")
    pred_values = load_pred_data(
        data_path=args.data_path,
        filename=args.filename
    )

    # Load "attentions" data
    print("\nLoading attentions data ...")
    attention_extra_data = load_attention_extra_data(
        data_path=args.data_path,
        filename=args.filename,
        attentions_to_load=attentions_to_load
    )

    # Merge all into a single dictionary
    print("\nPreparing full data ...")
    data = {}
    data.update(attributions_data)
    data.update(attention_extra_data)

    # Get attentions/attributions separetly from pred_values
    pred_values_attentions = {}
    pred_values_attributions = {}
    for k in pred_values:
        pred_values_attentions[k] = pred_values[k].pop("attentions")
        if "attributions" in pred_values[k]:
            pred_values_attributions[k] = pred_values[k].pop("attributions")
    data['attentions'] = pred_values_attentions
    if pred_values_attributions != {}:
        data['attributions'] = pred_values_attributions
    data['predictions'] = pred_values

    # If decompx classifier, get only the attributions of the correct label
    if "decompx_classifier" in techniques_to_load:
        original_decompx_data = data.pop("decompx_classifier")
        data_to_append = {}
        for version in existing_versions:
            data_to_append[version] = [d[:,l] for d, l in zip(original_decompx_data[version], data['predictions'][version]["labels"])]
        data['decompx_classifier'] = data_to_append

    print("\nCoherence checks ...")
    print(f"Data keys: {data.keys()}")
    # Assert all subdicts have the same keys, that should correspond to the versions of the models
    subdict_keys = [v.keys() for k, v in data.items()]
    assert all(k == subdict_keys[0] for k in subdict_keys)
    assert list(subdict_keys[0]) == existing_versions
    print(f"Model versions: {subdict_keys[0]}")

    # Print accuracies for coherence check
    accs = []
    f1s = []
    for ix in existing_versions:
        accs.append(accuracy_score(data["predictions"][ix]["labels"], data["predictions"][ix]["preds"]))
        f1s.append(f1_score(data["predictions"][ix]["labels"], data["predictions"][ix]["preds"], average='macro'))
    print(f"{accs}\nAccuracy: {100*np.mean(accs):.2f} +- {100*np.std(accs):.2f}")
    print(f"{f1s}\nF1-Macro: {100*np.mean(f1s):.2f} +- {100*np.std(f1s):.2f}")

    # Save output file
    save_path = os.path.join(args.data_path, f"all_analysis_data_{args.filename}.joblib")
    print(f"\nSaving to: {save_path}")
    joblib.dump(data, save_path, compress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("data_path", type=str)
    parser.add_argument("filename", type=str)
    parser.add_argument("techniques_to_load", type=str)
    parser.add_argument("attentions_to_load", type=str)

    args = parser.parse_args()

    main(args)
