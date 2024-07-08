import os
from argparse import ArgumentParser
from glob import glob

import joblib
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm

from src.approach import ModelModule
from src.modules import ModelAltiModule, ModelDecompxModule
from src.utils import load_hparams


def compute_joint_attention(att_mat, add_residual=True):
    # Adapted from and verified with:
    # https://github.com/samiraabnar/attention_flow/blob/8044f5312f4ced18d4cf66ffe28f6c045629b4ed/attention_graph_util.py#L104
    if add_residual:
        residual_att = torch.eye(
            att_mat.shape[-1], device=att_mat.device
        ).unsqueeze(0).repeat(att_mat.shape[1], 1, 1).unsqueeze(0)
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1).unsqueeze(-1)
    else:
       aug_att_mat = att_mat

    joint_attentions = torch.zeros_like(aug_att_mat)

    num_layers = joint_attentions.shape[1]
    joint_attentions[:, 0] = aug_att_mat[:, 0]
    for i in range(1, num_layers):
        joint_attentions[:, i] = torch.bmm(aug_att_mat[:, i], joint_attentions[:, i-1])

    return joint_attentions


def compute_attention_rollout(attentions, add_residual=True):
    # Compute mean over attention heads
    # attentions: [batch_size, num_layers, num_heads, num_tokens, num_tokens]
    # attentions_mean_heads: [batch_size, num_layers, num_tokens, num_tokens]
    attentions_mean_heads = attentions.mean(dim=2)

    # Compute joint attentions
    joint_attentions = compute_joint_attention(attentions_mean_heads, add_residual=add_residual)
    
    return joint_attentions


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
    print("Loading model from: ", model_path)
    if args.do_decompx:
        model_module = ModelDecompxModule
    elif args.do_alti:
        model_module = ModelAltiModule
    else:
        model_module = ModelModule
    model = model_module.load_from_checkpoint(model_path).to(device)
    model = model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    # List with num_examples values, with shape [num_layers, num_tokens, num_tokens]
    all_attention_rollout_values = []
    all_decompx_values = []
    all_decompx_classifier_values = []
    all_alti_values = []
    all_alti_aggregated_values = []
    preds = []
    for example_idx_start in tqdm(range(0, len(pred_data["input_ids"]), args.batch_size), miniters=100):
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

        # List with num_layers elements of shape [batch_size, num_heads, num_tokens, num_tokens]
        with torch.no_grad():
            model_output = model(
                inputs_embeds=model.model.bert.embeddings(examples),
                attention_mask=examples_mask,
                token_type_ids=token_type_ids
            )

        # attentions: [batch_size, num_layers, num_heads, num_tokens, num_tokens]
        if "attentions" in model_output:
            attentions = model_output["attentions"]
            attentions = torch.stack(attentions, dim=1)
        
        # hidden_states: [batch_size, 1 + num_layers, num_tokens, embed_dim]
        # first entry is the output of embedding layer
        if "hidden_states" in model_output:
            hidden_states = model_output["hidden_states"]
            hidden_states = torch.stack(hidden_states, dim=1)

        preds.extend(torch.argmax(model_output["pred_logits"], -1).tolist())

        # Assert attentions and logits are the same as in the pred data
        model_out_attn_assert = attentions.mean(dim=2)
        for i in range(model_out_attn_assert.shape[0]):
            pred_data_attn_assert = pred_data["attentions"][example_idx[i]].cpu()
            try:
                assert torch.all(torch.isclose(model_out_attn_assert[i, :, 0, :pred_data_attn_assert.shape[-1]].cpu(), pred_data_attn_assert, atol=1e-3))
            except:
                print("WARNING:\n", model_out_attn_assert[i, :, 0, :pred_data_attn_assert.shape[-1]].cpu() - pred_data_attn_assert)
        pred_data_logits_assert = pred_data["logits"][example_idx[0]:example_idx[i]+1]
        try:
            assert torch.all(torch.isclose(model_output["pred_logits"].cpu(), pred_data_logits_assert, atol=1e-3))
        except:
            print("WARNING:\n", model_output["pred_logits"].cpu() - pred_data_logits_assert)

        if args.do_attention_rollout:
            # Compute attention rollouts
            attention_rollout_values = compute_attention_rollout(attentions)
            for att_roll_values, msk  in zip(attention_rollout_values, examples_mask):
                masked_values = att_roll_values[:, :torch.sum(msk), :torch.sum(msk)]
                # Keep only CLS
                all_attention_rollout_values.append(masked_values[:, 0, :].cpu())

        if args.do_decompx:
            decompx_agg_all_layers = torch.stack(model_output["decompx_all_layers"].aggregated, dim=1)
            decompx_agg_all_layers = decompx_agg_all_layers[:, :, 0, :]
            for dcx_values, dcx_classifier_values, msk in zip(decompx_agg_all_layers, model_output["decompx_last_layer"].classifier, examples_mask):
                all_decompx_values.append(dcx_values[:, :torch.sum(msk)].cpu().clone())
                all_decompx_classifier_values.append(dcx_classifier_values[:torch.sum(msk)].cpu().clone())

        if args.do_alti:
            # ALTI is already masked - Keep only CLS
            alti_values = [x[:, 0].cpu().clone() for x in model_output["contributions"]]
            alti_values_agg = [x[:, 0].cpu().clone() for x in model_output["contributions_aggregated"]]
            all_alti_values.extend(alti_values)
            all_alti_aggregated_values.extend(alti_values_agg)

    # Assert predictions are the same
    assert pred_data['preds'].tolist() == preds
    if not torch.equal(pred_data['logits'][-model_output["pred_logits"].shape[0]:], model_output["pred_logits"].cpu()):
        assert torch.all(
            torch.isclose(pred_data['logits'][-model_output["pred_logits"].shape[0]:], model_output["pred_logits"].cpu(), atol=1e-4),
        )

    # Save data
    if args.do_attention_rollout:
        save_path = os.path.join(args.path, f"attention_rollout_{os.path.splitext(args.pred_path_filename)[0]}.joblib")
        print("Saving to: ", save_path)
        joblib.dump(all_attention_rollout_values, save_path, compress=4)

    if args.do_decompx:
        save_path = os.path.join(args.path, f"attributions_{os.path.splitext(args.pred_path_filename)[0]}_decompx.joblib")
        print("Saving to: ", save_path)
        joblib.dump(all_decompx_values, save_path, compress=4)

        save_path = os.path.join(args.path, f"attributions_{os.path.splitext(args.pred_path_filename)[0]}_decompx_classifier.joblib")
        print("Saving to: ", save_path)
        joblib.dump(all_decompx_classifier_values, save_path, compress=4)

    if args.do_alti:
        save_path = os.path.join(args.path, f"attributions_{os.path.splitext(args.pred_path_filename)[0]}_alti.joblib")
        print("Saving to: ", save_path)
        joblib.dump(all_alti_values, save_path, compress=4)

        save_path = os.path.join(args.path, f"attributions_{os.path.splitext(args.pred_path_filename)[0]}_alti_aggregated.joblib")
        print("Saving to: ", save_path)
        joblib.dump(all_alti_aggregated_values, save_path, compress=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--pred_path_filename", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--do_attention_rollout", action="store_true")
    parser.add_argument("--do_decompx", action="store_true")
    parser.add_argument("--do_alti", action="store_true")
    args = parser.parse_args()

    main(args)
