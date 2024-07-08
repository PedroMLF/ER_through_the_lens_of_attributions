import os
from argparse import ArgumentParser
from glob import glob

import captum
import joblib
import torch
from pytorch_lightning import seed_everything
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm

from src.approach import ModelModule
from src.utils import load_hparams


class IntegratedGradients:
    def __init__(self, model, forward_fn, baseline_vector, num_steps):
        self.baseline_vector = baseline_vector
        self.num_steps = num_steps
        self.model = model

        self.method = captum.attr.IntegratedGradients(forward_fn)

    def attribute(self, example, example_mask, token_type_ids, target_label):
        b, n = example.shape
        attributions, _ = self.method.attribute(
            inputs=self.model.model.bert.embeddings(example),
            baselines=self.baseline_vector.expand(-1, n, -1),
            additional_forward_args=(example_mask, token_type_ids),
            target=target_label,
            n_steps=self.num_steps,
            return_convergence_delta=True
        )
        return attributions


class InputxGradient:
    def __init__(self, model, forward_fn):
        self.model = model

        self.method = captum.attr.InputXGradient(forward_fn)

    def attribute(self, example, example_mask, token_type_ids, target_label):
        attributions = self.method.attribute(
            inputs=self.model.model.bert.embeddings(example),
            additional_forward_args=(example_mask, token_type_ids),
            target=target_label,
        )
        return attributions


def normalize_attributions(attributions, mask):
    attributions_sum = attributions.sum(-1).squeeze()
    attributions_sum /= torch.norm(attributions_sum, dim=-1).unsqueeze(-1)
    attributions_sum_masked = []
    for att, msk in zip(attributions_sum, mask):
        attributions_sum_masked.append(att[msk==1].detach().cpu())
    return attributions_sum_masked


def main(args):

    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get hparams to get embedding model
    hparams_path = os.path.join(args.path, "hparams.yaml")
    hparams = load_hparams(hparams_path)

    embedding_model = hparams["model_name"]

    # Load data and model
    pred_path = glob(os.path.join(args.path, args.pred_path_filename))
    assert len(pred_path) == 1, print(pred_path)
    pred_path = pred_path[0]
    pred_data = joblib.load(pred_path)

    model_path = glob(os.path.join(args.path, "*.ckpt"))
    assert len(model_path) == 1, print(model_path)
    model_path = model_path[0]
    model = ModelModule.load_from_checkpoint(model_path).to(device)
    model = model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    # Define the interpretability method
    if args.technique == "IG":
        interpretability_technique = IntegratedGradients(
            model=model,
            forward_fn=model.forward_captum,
            baseline_vector=model.model.bert.embeddings(torch.tensor(tokenizer.pad_token_id).reshape(1, 1).to(device)),
            num_steps=100,
        )
    elif args.technique == "IxG":
        interpretability_technique = InputxGradient(
            model=model,
            forward_fn=model.forward_captum,
        )

    weights = {"attributions": []}

    for example_idx_start in tqdm(range(0, len(pred_data["input_ids"]), args.batch_size), miniters=250):
        # Get indices of multiple examples in batch
        example_idx = list(range(example_idx_start, example_idx_start + args.batch_size))
        # Get all target labels
        target_labels = pred_data['labels'][example_idx[0] : example_idx[-1] + 1].to(device)
        # Build padded example ids
        examples = pred_data['input_ids'][example_idx[0] : example_idx[-1] + 1]
        examples = pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        # Build examples mask
        examples_mask = (examples != tokenizer.pad_token_id).int().to(device)
        # Build token type ids
        token_type_ids = torch.zeros_like(examples)
        sep_ids = torch.argmax(torch.eq(examples, tokenizer.sep_token_id).int(), dim=1)
        for row, sep_id in zip(token_type_ids, sep_ids):
            row[sep_id+1:] = 1
        token_type_ids = token_type_ids * examples_mask

        # Get attributions
        attributions = interpretability_technique.attribute(
            example=examples,
            example_mask=examples_mask,
            token_type_ids=token_type_ids,
            target_label=target_labels
        )
        attributions_sum = normalize_attributions(attributions, examples_mask)

        weights["attributions"].extend(attributions_sum)

    # Save data
    save_path = os.path.join(args.path, f"attributions_{os.path.splitext(args.pred_path_filename)[0]}_{args.technique}.joblib")
    print("Saving to: ", save_path)
    joblib.dump(weights, save_path, compress=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--pred_path_filename", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--technique", type=str, required=True)
    args = parser.parse_args()

    assert args.technique in ["IG", "IxG"]

    main(args)
