import math
import os
import random
import string
import time
from argparse import ArgumentParser
from collections import Counter
from glob import glob

import joblib
import numpy as np
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm
from transformers import AutoTokenizer
from unidecode import unidecode


def joblib_save(data, path):
    print(f"Saving data ({len(data)} examples) to: ", path)
    joblib.dump(data, path)


def joblib_load(path):
    return joblib.load(path)


class SAPreprocessor:
    def __init__(self, args):
        print("Initializing SA preprocessor...")
        self.max_sequence_length = args.max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess_example(self, example):

        create_annotation_targets = "rationale" in example.keys()

        tokenized_dataset = self.tokenizer(
            example["text"],
            add_special_tokens=True,
            truncation=False,
            return_token_type_ids=True,
            return_special_tokens_mask=True
        )

        # ADAPTED FROM: https://github1s.com/INK-USC/ER-Test/blob/HEAD/scripts/build_dataset.py#L72-L73
        if create_annotation_targets:
            assert len(example["text"].split()) == len(example["rationale"])

            raw_tokens = example["text"].lower().split()
            tok_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_dataset["input_ids"], skip_special_tokens=True)
            raw_rationale = example["rationale"]

            annotation_targets = []
            j = 0
            cur_token = tok_tokens[j]

            for i in range(len(raw_tokens)):
                cur_raw_token = raw_tokens[i]
                cur_raw_rationale = raw_rationale[i]
                cur_reconstructed_raw_token = ''

                while len(cur_raw_token) > 0:
                    for char in cur_token:
                        # Fix ascii cases, otherwise cannot be compared
                        if char.isascii() and not cur_raw_token[0].isascii():
                            matches = (char == unidecode(cur_raw_token[0]))
                        else:
                            matches = (char == cur_raw_token[0])
                        #if char == cur_raw_token[0]:
                        if matches:
                            cur_raw_token = cur_raw_token[1:]
                            cur_reconstructed_raw_token += char

                    annotation_targets.append(cur_raw_rationale)
                    j += 1
                    cur_token = tok_tokens[j] if j < len(tok_tokens) else None

                assert unidecode(cur_reconstructed_raw_token) == unidecode(raw_tokens[i]), breakpoint()

            # We are doing classification, so this should be from [CLS] at the start and [SEP] at the end
            assert len(tokenized_dataset["input_ids"]) - len(annotation_targets)== 2

            annotation_targets = [0] + annotation_targets + [0]

            if len(annotation_targets) != len(tokenized_dataset["input_ids"]):
                raise Exception(
                    f"Length mismatch between annotation_targets ({len(annotation_targets)})\
                        and input_ids ({len(tokenized_dataset['input_ids'])})"
                )

            # All examples are used for training
            annotation_keep_loss = 1 if sum(annotation_targets) > 0 else 0

        batch = {
            "input_ids": tokenized_dataset["input_ids"],
            "attention_mask": tokenized_dataset["attention_mask"],
            "token_type_ids": tokenized_dataset["token_type_ids"],
            "special_tokens_mask": tokenized_dataset["special_tokens_mask"],
            "labels": example["label"],
        }

        if create_annotation_targets:
            batch["annotation_targets"] = annotation_targets
            batch["annotation_keep_loss"] = annotation_keep_loss

            # Coherence check - make sure rationales match the expected
            orig_align = [(x,y) for x,y in zip(example["text"].split(), example["rationale"])]
            orig_align_s = ' '.join([x for x,y in orig_align if y])

            obtained_align = [(x,y) for x,y in zip(self.tokenizer.convert_ids_to_tokens(batch["input_ids"]), annotation_targets)]
            obtained_align_s = ' '.join([x for x,y in obtained_align if y])

            # Coherence check - Workaround to join subwords -- do it only if there are actually rationales
            if annotation_keep_loss:
                if args.model == "bert-base-uncased":
                    fixed_obtained_align_s = [obtained_align_s.split()[0]]
                    for t in obtained_align_s.split()[1:]:
                        if t.startswith("##"):
                            fixed_obtained_align_s[-1] += t[2:]
                        else:
                            fixed_obtained_align_s.append(t)
                    fixed_obtained_align_s = ' '.join(fixed_obtained_align_s)
                elif args.model == "google/bigbird-roberta-base":
                    fixed_obtained_align_s = obtained_align_s.replace("▁", "")
                else:
                    raise NotImplementedError
            else:
                fixed_obtained_align_s = obtained_align_s

            # Coherence check - Some whitespaces might differ, e.g. things like "x-y" vs "x - y", but as long as they captured rationales that's not a problem
            assert unidecode(''.join(orig_align_s.split())) == unidecode(''.join(fixed_obtained_align_s.split()))

        # Apply max length
        for k in batch:
            if isinstance(batch[k], list):
                if len(batch[k]) > args.max_sequence_length:
                    batch[k] = batch[k][:args.max_sequence_length-1] + [batch[k][-1]]

        return batch


def main(args):

    seed_everything(42)

    preprocessor = SAPreprocessor(args)

    if args.filepath.split(".")[-1] != "joblib":
        raise Exception("Expected .joblib file")
    paths = [args.filepath]

    if not paths:
        raise Exception("No paths available.")

    for path in paths:
        print("Loading data from:", path)
        dataset = joblib.load(path)

        all_examples_preprocessed = {}

        for example_ix, example in tqdm(dataset.items(), desc="Preprocessing: ", miniters=2500):
            example_preprocessed = preprocessor.preprocess_example(example)
            if example_preprocessed:
                all_examples_preprocessed[len(all_examples_preprocessed)] = example_preprocessed

        # Split the data into dictionaries with args.data_step examples
        data_splits = []
        for i in range(math.ceil(len(all_examples_preprocessed)/args.data_step)):
            data_splits.append({k:v for k,v in all_examples_preprocessed.items() if k >= i*args.data_step and k < (i+1)*args.data_step})

        # Save chunked datasets
        save_path_suffix = "preprocessed"

        st = time.time()
        if len(data_splits) > 1:
            joblib.Parallel(n_jobs=len(data_splits))(
                joblib.delayed(joblib_save)(ds, f"{os.path.splitext(path)[0]}_{save_path_suffix}_{index}.joblib")
                for index, ds in enumerate(data_splits)
            )
        elif len(data_splits) == 1:
            joblib_save(data_splits[0], path=f"{os.path.splitext(path)[0]}_{save_path_suffix}.joblib")
        else:
            raise Exception
        print(f"{time.time()-st:.2f} second(s)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--data_step", type=int, default=100000)
    parser.add_argument("--model", type=str, default="google/bigbird-roberta-base")
    args = parser.parse_args()

    print("ARGS:")
    print(args)
    print()

    main(args)
