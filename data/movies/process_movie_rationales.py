import os
import re

import joblib
from datasets import load_dataset


def save_data(data, path):
    print("Saving data object...")
    joblib.dump(data, filename=path)
    print("Saved to: ", path)
    print("Number of items: ", len(data))


def main(dataset_name, dataset_splits):

    data = {}

    for split in dataset_splits:
        print("-" * 40)
        print(f"Processing: {dataset_name}/{split}")
        dataset = load_dataset(dataset_name, split=split)

        for example_ix, example in enumerate(dataset):
            # Prepare input
            text = example['review']

            tokens = text.split()

            rationales = [0] * len(tokens)

            for evidence in example['evidences']:
                evidence_tokens = evidence.split()

                # Iterate through the words in the original string
                matches = 0
                for i in range(len(tokens) - len(evidence_tokens) + 1):
                    # Check if the current substring in the original string matches the substring in the annot list
                    if tokens[i:i+len(evidence_tokens)] == evidence_tokens:
                        # Set the corresponding elements in rationales to 1
                        rationales[i:i+len(evidence_tokens)] = [1] * len(evidence_tokens)
                        matches += 1
                if not matches:
                    raise Exception()

            # Replace tokens
            text = ' '.join(tokens)
            text = text.replace("\x12", "\'")
            text = text.replace("\x05", "")

            data[len(data)] = {
                'label': example['label'],
                'text': text,
                'rationale': rationales,
            }

    save_data(data, os.path.join("data", "movies", "movies_dev-test_orig.joblib"))


if __name__ == "__main__":
    main(dataset_name="movie_rationales", dataset_splits=['validation', 'test'])
