import os
import random
import re
from collections import Counter

import datasets 
import joblib
from datasets import load_dataset
datasets.logging.set_verbosity_error()
from tqdm import tqdm


def rating_to_label(rating):
    return int(rating >=4)


def main():

    min_tokens = 10
    max_tokens = 1000
    num_examples = 5000

    for split in ['Movies_and_TV']:

        data = {}

        all_texts = []
        num_duplicates = 0
        num_short = 0
        num_long = 0

        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_Movies_and_TV",
            streaming=True,
        )
        shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)
        sampled_dataset = list(shuffled_dataset['full'].take(10000))

        for example in sampled_dataset:

            # We ignore the title
            text = example['text']
            text = text.replace('<br />', ' ')
            text = text.replace('&#34;', ' ')
            text = text.replace('&quot;', ' ')
            patterns = [r'&#\d{4};', r'https?://\S+']
            for pattern in patterns:
                text = re.sub(pattern, ' ', text)
            text = text.lower()

            # Get binary label and text into dict
            data_to_append = {
                'label': rating_to_label(example['rating']),
                'text': text,
            }

            # Append if text is not repeated
            if len(text.split()) < min_tokens:
                num_short += 1
            elif len(text.split()) > max_tokens:
                num_long += 1
            elif text in all_texts:
                num_duplicates += 1
            else:
                data[len(data)] = data_to_append
                all_texts.append(text)

            if len(data) == num_examples:
                break

        print(f"Skipped {num_duplicates} duplicates")
        print(f"Skipped {num_short} short reviews")
        print(f"Skipped {num_long} long reviews")
        print(f"Labels: {Counter([x['label'] for x in data.values()])}")
        print(f"Num examples: {len(data)}")
        print(f"Most common (to catch issues): {Counter([x for x in all_texts for x in x.split()]).most_common(50)}")
        for i in random.sample(list(range(len(data))), k=10):
            print(f">>> {data[i]}")

        # Save joblib
        print("Saving data object...")
        target_filepath = os.path.join("data", "amazon", f"amazon_movies-tv_test_orig.joblib")
        joblib.dump(data, filename=target_filepath)
        print("Saved to: ", target_filepath)


if __name__ == "__main__":
    main()