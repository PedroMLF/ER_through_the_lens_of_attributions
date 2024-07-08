import json
import os

from tqdm import tqdm
import joblib


def main():
    
    label_map = {'pos': 1, 'neg': 0}

    for split in ['train', 'dev', 'test']:

        data = json.load(open(f"data/sst/sst_{split}.json"))

        data_dict = {}
        for example_id, data_example in tqdm(enumerate(data), miniters=250):
            # The original example_id is not contiguous, as it skips neutral labels
            data_dict[example_id] = {
                'text': data_example['text'],
                'label': label_map[data_example['classification']],
                'rationale': [1 if x >= 0.5 else 0 for x in data_example['rationale']],
            }

            # Fix similar to: https://github.com/INK-USC/ER-Test/blob/main/scripts/build_dataset.py#L425
            if len(data_example['rationale']) != len(data_example['text'].split()):
                len_diff = len(data_example['text'].split()) - len(data_example['rationale'])
                if len_diff < 0:
                    raise Exception
                else:
                    print(f"\n-- Fixing example {example_id} by adding {len_diff} zero(s) to rationales")
                    new_rationale = data_dict[example_id]['rationale'] + [0] * len_diff
                    data_dict[example_id]['rationale'] = new_rationale

        save_path = os.path.join(f"data/sst/sst_{split}_orig.joblib")
        print(f"Saving {len(data_dict)} example(s) to {save_path}")
        joblib.dump(data_dict, save_path, compress=True)


if __name__ == "__main__":
    main()
