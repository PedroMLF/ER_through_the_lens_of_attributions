import os
from argparse import ArgumentParser
from glob import glob

import joblib
import numpy as np
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm

from src.approach import ModelModule
from src.utils import load_hparams


def get_top_ratio_attributions(scores, ratio):
    rationales = []
    for example_scores in scores:
        k = max(1, int(ratio * len(example_scores)))
        top_k_indices = torch.topk(example_scores, k=k).indices
        example_rationales = torch.zeros_like(example_scores).int()
        example_rationales[top_k_indices] = 1
        rationales.append(example_rationales)
    return rationales


def get_logits(model, pred_data, batch_size, num_examples, device, pad_token_id, sep_token_id, input_ids=None, use_zero_vector=False):

    logits = []

    for example_idx_start in tqdm(range(0, num_examples, batch_size), miniters=500, desc="Computing predictions:"):
        # Get indices of multiple examples in batch
        example_idx = list(range(example_idx_start, example_idx_start + batch_size))
        # Build padded example ids
        all_input_ids = input_ids if input_ids else pred_data['input_ids']
        examples = all_input_ids[example_idx[0] : example_idx[-1] + 1]
        examples = pad_sequence(examples, batch_first=True, padding_value=pad_token_id).to(device)
        # Build examples mask
        examples_mask = (examples.cpu() != pad_token_id).int().to(device)
        # Build token type ids
        token_type_ids = torch.zeros_like(examples)
        sep_ids = torch.argmax(torch.eq(examples, sep_token_id).int(), dim=1)
        for row, sep_id in zip(token_type_ids, sep_ids):
            row[sep_id+1:] = 1
        token_type_ids = token_type_ids * examples_mask

        # List with num_layers elements of shape [batch_size, num_heads, num_tokens, num_tokens]
        with torch.no_grad():
            if use_zero_vector:
                inputs_embeds = torch.zeros(
                    (examples.shape[0], examples.shape[1], model.model.config.hidden_size),
                    device=device
                )
            else:
                inputs_embeds = model.model.bert.embeddings(examples)

            model_output = model(inputs_embeds=inputs_embeds, attention_mask=examples_mask, token_type_ids=token_type_ids)

        #predictions.extend(torch.argmax(model_output["pred_logits"],dim=-1).tolist())
        logits.extend(model_output["pred_logits"])
    
    logits = torch.stack(logits)

    return logits


def compute_sufficiency(preds, pred_logits, suff_logits):
    pred_probs = torch.nn.functional.softmax(pred_logits, dim=-1)
    suff_probs = torch.nn.functional.softmax(suff_logits, dim=-1)

    probs_diff = pred_probs[torch.arange(len(preds)), preds] - suff_probs[torch.arange(len(preds)), preds]
    probs_diff_clamped = torch.clamp(probs_diff, min=0)
    suff = 1 - probs_diff_clamped

    return suff, probs_diff


def compute_norm_sufficiency(suff, suff_zero):
    suff_zero -= 1e-4
    norm_suff = (suff - suff_zero) / (1 - suff_zero)
    norm_suff = torch.clamp(norm_suff, min=0, max=1)
    return norm_suff


def compute_comprehensiveness(preds, pred_logits, comp_logits):
    pred_probs = torch.nn.functional.softmax(pred_logits, dim=-1)
    comp_probs = torch.nn.functional.softmax(comp_logits, dim=-1)

    probs_diff = pred_probs[torch.arange(len(preds)), preds] - comp_probs[torch.arange(len(preds)), preds]
    comp = torch.clamp(probs_diff, min=0)

    return comp, probs_diff


def compute_norm_comprehensiveness(comp, suff_zero):
    suff_zero -= 1e-4
    norm_comp = comp / (1 - suff_zero)
    norm_comp = torch.clamp(norm_comp, min=0, max=1)
    return norm_comp


def main(args):

    print(f"\nARGS: {args}\n")

    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all the pre-computed attributions
    all_analysis_data = joblib.load(os.path.join(args.path, f"all_analysis_data_{args.pred_filename}.joblib"))

    # Iterate over all paths and compute the necessary values
    all_paths = sorted(glob(os.path.join(args.path, "version_*")), key=lambda x: int(x.split("version_")[-1]))
    all_values = {}
    for path in all_paths:

        print(f"\n\nPATH: {path}\n\n")

        model_version = int(path.split("version_")[-1])

        # Get hparams to get embedding model
        hparams_path = os.path.join(path, "hparams.yaml")
        hparams = load_hparams(hparams_path)

        embedding_model = hparams["model_name"]

        # Load data and model
        pred_path = glob(os.path.join(path, f"{args.pred_filename}.joblib"))
        assert len(pred_path) == 1, print(pred_path)
        pred_path = pred_path[0]
        pred_data = joblib.load(pred_path)

        model_path = glob(os.path.join(path, "*.ckpt"))
        assert len(model_path) == 1, print(model_path)
        model_path = model_path[0]
        print("Loading model from: ", model_path)
        model = ModelModule.load_from_checkpoint(model_path).to(device)
        model = model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(embedding_model, use_fast=False)

        num_examples = len(pred_data["input_ids"])
        values = {}

        for technique in args.techniques.split(","):
            print(f"\n--- {technique} ---\n")
            values[technique] = {}

            # Get attributions
            if technique == "random":
                attributions = [torch.randn(ex.shape) for ex in pred_data["input_ids"]]
            elif technique == "human":
                pass
            else:
                model_version = int(path.split("version_")[-1].strip(os.path.sep))
                attributions = all_analysis_data[technique][model_version]

                # Some techniques have multiple layers, we select the top-layer
                if len(attributions[0].shape) == 2:
                    assert attributions[0].shape[0] == model.model.config.num_hidden_layers
                    attributions = [att[-1] for att in attributions]

            ratios = [0.01, 0.20, 0.40, 0.60, 0.80, 1.00] if technique != "human" else [0]
            for ratio in ratios:
                values[technique][ratio] = {}
                # Get top attributions
                if technique == "human":
                    rationales = pred_data["annotation_targets"]
                else:
                    rationales = get_top_ratio_attributions(attributions, ratio=ratio)

                # Prepare sufficiency data
                sufficiency_input_ids = []
                for example_ix in tqdm(range(num_examples), desc=f"{technique} - Sufficiency Data - Ratio {ratio}: ", miniters=1000):
                    special_tokens_mask = torch.tensor(tokenizer.get_special_tokens_mask(pred_data["input_ids"][example_ix], already_has_special_tokens=True))
                    # Get both the special tokens and the human annotated words to keep
                    ixs_to_keep = torch.logical_or(special_tokens_mask, rationales[example_ix]).nonzero().squeeze(-1)
                    sufficiency_input_ids.append(pred_data["input_ids"][example_ix][ixs_to_keep])

                # Prepare comprehensiveness data
                comprehensiveness_input_ids = []
                for example_ix in tqdm(range(num_examples), desc=f"{technique} - Comprehensiveness Data - Ratio {ratio}: ", miniters=1000):
                    special_tokens_mask = torch.tensor(tokenizer.get_special_tokens_mask(pred_data["input_ids"][example_ix], already_has_special_tokens=True))
                    # Get both the special tokens and the human non-annotated words to keep
                    ixs_to_keep = torch.logical_or(special_tokens_mask, rationales[example_ix] == 0).nonzero().squeeze(-1)
                    comprehensiveness_input_ids.append(pred_data["input_ids"][example_ix][ixs_to_keep])

                # Compute all necessary logits
                zero_vector_logits = get_logits(
                    model=model,
                    pred_data=pred_data,
                    num_examples=num_examples,
                    batch_size=args.batch_size,
                    input_ids=sufficiency_input_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    sep_token_id=tokenizer.sep_token_id,
                    device=device,
                    use_zero_vector=True,
                )

                suff_logits = get_logits(
                    model=model,
                    pred_data=pred_data,
                    num_examples=num_examples,
                    batch_size=args.batch_size,
                    input_ids=sufficiency_input_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    sep_token_id=tokenizer.sep_token_id,
                    device=device,
                )

                comp_logits = get_logits(
                    model=model,
                    pred_data=pred_data,
                    num_examples=num_examples,
                    batch_size=args.batch_size,
                    input_ids=comprehensiveness_input_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    sep_token_id=tokenizer.sep_token_id,
                    device=device,
                )

                # Compute scores
                suff, suff_raw = compute_sufficiency(preds=pred_data["preds"], pred_logits=pred_data["logits"], suff_logits=suff_logits.cpu())
                zero_suff, _ = compute_sufficiency(preds=pred_data["preds"], pred_logits=pred_data["logits"], suff_logits=zero_vector_logits.cpu())
                norm_suff = compute_norm_sufficiency(suff, zero_suff)
                values[technique][ratio]["sufficiency"] = norm_suff
                values[technique][ratio]["sufficiency_raw"] = suff_raw

                comp, comp_raw = compute_comprehensiveness(preds=pred_data["preds"], pred_logits=pred_data["logits"], comp_logits=comp_logits.cpu())
                norm_comp = compute_norm_comprehensiveness(comp, zero_suff)
                values[technique][ratio]["comprehensiveness"] = norm_comp
                values[technique][ratio]["comprehensiveness_raw"] = comp_raw

        # Add current values to all_values dict
        all_values[model_version] = values

    joblib.dump(all_values, os.path.join(args.path, f"all_suff_comp_{args.pred_filename}_spaced.joblib"), compress=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--pred_filename", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--techniques", type=str, default="attentions,rollout,IxG,alti_aggregated,decompx,decompx_classifier,random,human")
    args = parser.parse_args()

    main(args)

# python compute_suff_comp_metrics.py --path checkpoints/baseline/train/ --pred_filename pred_sst_dev_data --batch_size 4 --techniques attentions