import os
import random
from collections import Counter

import joblib
from datasets import load_dataset


def main(dataset_name, dataset_splits):
    for split in dataset_splits:
        random.seed(42)
        print("-" * 40)
        print(f"Processing: {dataset_name}/{split}")
        dataset = load_dataset(dataset_name, split=split)

        # We sample 5k examples
        data = {}
        for example_ix in random.sample(list(range(len(dataset))), k=5000):

            text = dataset[example_ix]['text']
            text = text.replace('<br />', ' ')
            text = text.replace('&#34;', ' ')
            text = text.replace('&quot;', ' ')
            text = text.lower()

            data[len(data)] = {
                'label': dataset[example_ix]['label'],
                'text': text,
            }

        # Print stats
        print(Counter((x['label'] for x in data.values())))
        print(f"Most common (to catch issues): {Counter([x for x in data for x in data[x]['text'].split()]).most_common(100)}")

        # Save joblib
        print("Saving data object...")
        target_filepath = os.path.join("data", "imdb", f"imdb_{split}_orig.joblib")
        joblib.dump(data, filename=target_filepath)
        print("Saved to: ", target_filepath)
        print("Number of items: ", len(data))


if __name__ == "__main__":
    main(dataset_name="stanfordnlp/imdb", dataset_splits=['test'])
