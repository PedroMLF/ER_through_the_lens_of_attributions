import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from argparse import ArgumentParser
from collections import defaultdict, Counter
from functools import partial
from glob import glob

import numpy as np
import joblib
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from src.modeling_big_bird import BigBirdForSequenceClassification
from src.trainer import Trainer
from src.utils import print_sep


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        # Read variables from args
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.dataloader_num_workers = args.dataloader_num_workers

        # Create data paths
        self.train_data_paths = glob(os.path.join(args.data_dir, args.train_data_filename))
        self.val_data_paths = glob(os.path.join(args.data_dir, args.dev_data_filename))
        self.test_data_paths = glob(os.path.join(args.data_dir, args.test_data_filename))
        self.predict_data_paths = glob(os.path.join(args.data_dir, args.predict_data_filename))

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("data")

        parser.add_argument("--data_dir", type=str, required=True)
        parser.add_argument("--dataloader_num_workers", type=int, default=2)
        parser.add_argument("--train_data_filename", type=str, default="train.txt")
        parser.add_argument("--dev_data_filename", type=str, default="dev.txt")
        parser.add_argument("--predict_data_filename", type=str, default="predict.txt")
        parser.add_argument("--test_data_filename", type=str, default="test.txt")
        parser.add_argument("--train_batch_size", type=int, default=16)
        parser.add_argument("--eval_batch_size", type=int, default=16)

        return parent_parser

    def prepare_data(self, stage=None):
        if stage == "fit":
            self.train_data = self._load_dataset(self.train_data_paths)
            self.val_data = self._load_dataset(self.val_data_paths)
        elif stage == "eval":
            self.test_data = self._load_dataset(self.test_data_paths)
        elif stage == "predict":
            self.predict_data = self._load_dataset(self.predict_data_paths)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self._get_dataloader(self.train_data, is_train=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_data, is_train=False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_data, is_train=False)

    def predict_dataloader(self):
        return self._get_dataloader(self.predict_data, is_train=False)

    def _get_dataloader(self, dataset, is_train):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_batch_size if is_train else self.eval_batch_size,
            collate_fn=partial(self._collate_fn, is_train=is_train),
            num_workers=self.dataloader_num_workers,
            shuffle=is_train,
            pin_memory=True,
        )

    def _collate_fn(self, dataset, is_train):

        create_annotation_targets = "annotation_targets" in dataset[0].keys()

        # Group relevant variables
        input_ids = []
        attention_mask = []
        token_type_ids = []
        special_tokens_mask = []
        labels = []
        if create_annotation_targets:
            annotation_targets = []
            annotation_keep_loss = []

        for ds_example in dataset:
            input_ids.append(torch.tensor(ds_example["input_ids"]))
            attention_mask.append(torch.tensor(ds_example["attention_mask"]))
            token_type_ids.append(torch.tensor(ds_example["token_type_ids"]))
            special_tokens_mask.append(torch.tensor(ds_example["special_tokens_mask"]))
            labels.append(ds_example["labels"])
            if create_annotation_targets:
                annotation_targets.append(torch.tensor(ds_example["annotation_targets"]))
                annotation_keep_loss.append(ds_example["annotation_keep_loss"])

        # Pad tensors
        batch_tensor = {
            "input_ids": pad_sequence(input_ids, batch_first=True),
            "attention_mask": pad_sequence(attention_mask, batch_first=True),
            "labels": torch.tensor(labels).squeeze(),
            "token_type_ids": pad_sequence(token_type_ids, batch_first=True),
            "special_tokens_mask": pad_sequence(special_tokens_mask, batch_first=True),
        }

        if create_annotation_targets:
            batch_tensor["annotation_targets"] = pad_sequence(annotation_targets, batch_first=True)
            batch_tensor["annotation_keep_loss"] = torch.tensor(annotation_keep_loss).squeeze()

        return batch_tensor

    def _load_single_dataset(self, path):
        if os.path.exists(path):
            print("Loading data from: ", path)
            return joblib.load(path)
        else:
            raise Exception(f"Data path: {path} does not exist.")

    def _load_dataset(self, paths):
        st = time.time()
        if len(paths) == 1:
            dataset = self._load_single_dataset(paths[0])
        elif len(paths) > 1:
            loaded_data = joblib.Parallel(n_jobs=max(len(paths),8))(joblib.delayed(self._load_single_dataset)(path) for path in paths)
            dataset = {}
            for ds in loaded_data:
                dataset.update(ds)
            if list(range(max(dataset.keys())+1)) != sorted(list(dataset.keys())):
                raise Exception("Loaded data is missing keys")
        else:
            raise Exception(f"No paths.")
        print(f"Loading took: {time.time()-st:.2f} second(s)")
        return dataset


class BaseModelModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()

        # Read args
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.scheduler = args.scheduler
        self.constrained_optimization = args.constrained_optimization
        self.constrained_optimization_bound_init = args.constrained_optimization_bound_init
        self.constrained_optimization_bound_min = args.constrained_optimization_bound_min
        self.constrained_optimization_validation_bound = args.constrained_optimization_validation_bound
        self.constrained_optimization_loss = args.constrained_optimization_loss
        self.constrained_optimization_smoothing = args.constrained_optimization_smoothing
        self.constrained_optimization_lr = args.constrained_optimization_lr
        self.use_annotation_loss_only = args.use_annotation_loss_only

        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Disable automatic optimization if constrained optimization
        if self.constrained_optimization:
            self.automatic_optimization=False
            self.alpha_cl = torch.nn.Parameter(torch.ones(()))
            self.alpha_cl.register_hook(lambda grad: -grad)
            self.register_buffer("margin_cl", torch.tensor(self.constrained_optimization_bound_init))
            self.metric_to_track = args.metric_to_track

    @staticmethod
    def add_base_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")

        parser.add_argument("--learning_rate", type=float, default=5e-4)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--scheduler", type=str, default="linear")
        parser.add_argument("--constrained_optimization", action="store_true")
        parser.add_argument("--constrained_optimization_bound_init", type=float)
        parser.add_argument("--constrained_optimization_bound_min", type=float)
        parser.add_argument("--constrained_optimization_validation_bound", type=float)
        parser.add_argument("--constrained_optimization_smoothing", type=float)
        parser.add_argument("--constrained_optimization_loss", type=str)
        parser.add_argument("--constrained_optimization_lr", type=float)
        parser.add_argument("--use_annotation_loss_only", action="store_true")

        return parent_parser

    @staticmethod
    def check_base_model_specific_args(args):
        if args.scheduler not in ["linear", "reduce_lr_on_plateau", "None"]:
            raise Exception(f"--scheduler \"{args.scheduler}\" not defined")

        if args.constrained_optimization:
            if args.constrained_optimization_loss not in ["classifier", "guided"]:
                raise Exception(f"--constrained_optimization loss '{args.constrained_optimization_loss}' not defined")

            if args.use_annotation_loss_only:
                raise Exception(f"--use_annotation_loss_only not compatible with --constrained_optimization")

    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        if not self.automatic_optimization:
            # Apply before zero_grad
            self.alpha_cl.data = torch.where(self.alpha_cl.data < 0, torch.full_like(self.alpha_cl.data, 0), self.alpha_cl.data)

            opts = self.optimizers()
            for opt in opts:
                opt.zero_grad()

        output = self._step(batch, is_train=True)
        self.log("loss_train", output["loss"])
        if "loss_ce" in output:
            self.log("loss_ce", output["loss_ce"])
        if "loss_annotation" in output:
            self.log("loss_annotation", output["loss_annotation"])
        if "loss_constrained" in output:
            self.log("loss_constrained", output["loss_constrained"])
            self.log("loss_constrained_multiplier", self.alpha_cl.item())            
            self.log("loss_constrained_bound", self.margin_cl.item())
            if self.constrained_optimization_loss == "classifier":
                key = "ce"
            elif self.constrained_optimization_loss == "guided":
                key = "annotation"
            self.log("loss_bounded_difference", output[f"loss_{key}"] - self.margin_cl.item())
        if "entropy_attributions" in output:
            self.log("entropy_attributions", output["entropy_attributions"])

        if not self.automatic_optimization:
            self.manual_backward(output["loss"])
            for opt in opts:
                opt.step()

            # clip gradients
            #if args.gradient_clip_val:
            #    self.clip_gradients(opts[0], gradient_clip_val=args.gradient_clip_val)
            #    self.clip_gradients(opts[1], gradient_clip_val=args.gradient_clip_val)

            # call scheduler
            if self.scheduler == "reduce_lr_on_plateau":
                raise NotImplementedError

            schs = self.lr_schedulers()

            if isinstance(schs, list):
                for sch in schs:
                    sch.step()
            else:
                schs.step()

        return output

    def training_epoch_end(self, outputs):
        if self.constrained_optimization:
            if self.constrained_optimization_loss == "classifier":
                key = "ce"
            elif self.constrained_optimization_loss == "guided":
                key = "annotation"

            avg_loss_bounded = sum([o[f"loss_{key}"] for o in outputs]) / len(outputs)
            if avg_loss_bounded < 1.1 * self.margin_cl.item() and self.constrained_optimization_smoothing != 1.0:
                self.margin_cl = max(self.margin_cl * self.constrained_optimization_smoothing, torch.tensor(self.constrained_optimization_bound_min))
                print(f"New loss constraint bound: {self.margin_cl:.4f}")

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, is_train=False)
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = sum([o["loss"] for o in outputs]) / len(outputs)
        self.log("val_avg_loss", avg_loss.item())

        if "loss_ce" in outputs[0]:
            avg_loss_ce = sum([o["loss_ce"] for o in outputs]) / len(outputs)
            self.log("val_avg_loss_ce", avg_loss_ce.item())
        if "loss_annotation" in outputs[0]:
            avg_loss_annotation = sum([o["loss_annotation"] for o in outputs]) / len(outputs)
            self.log("val_avg_loss_annotation", avg_loss_annotation.item())
        if "loss_constrained" in outputs[0]:
            avg_loss_constrained = sum([o["loss_constrained"] for o in outputs]) / len(outputs)
            self.log("val_avg_loss_constrained", avg_loss_constrained.item())

        # Calculate metrics
        metrics = self.compute_metrics(outputs)
        self.log("val_f1_score", metrics['f1_macro'])
        self.log("val_acc_score", metrics['acc'])

    def _update_early_stopping_callback(self, new_score):
        self.trainer.early_stopping_callback.wait_count = 0
        self.trainer.early_stopping_callback.best_score = torch.tensor(new_score)

    def on_validation_end(self):
        if self.constrained_optimization:
            # Get the correct bounded loss
            if self.constrained_optimization_loss == "guided":
                loss_key = "val_avg_loss_annotation"
            elif self.constrained_optimization_loss == "classifier":
                loss_key = "val_avg_loss_ce"
            avg_loss = self.trainer.callback_metrics.get(f"{loss_key}").item()

            if avg_loss < 1.1 * self.constrained_optimization_validation_bound:
                existing_ckpts = glob(os.path.join(self.trainer.log_dir, "*.ckpt"))
                save = False

                new_score = self.trainer.callback_metrics.get(f"val_{self.metric_to_track}").item()

                if not existing_ckpts:
                    save = True
                    print(f"Saving checkpoint based on meeting constrained optimization bound at epoch: {self.trainer.current_epoch} - {self.metric_to_track} with score {new_score:.5f}")
                    self._update_early_stopping_callback(new_score)

                    # We use larger patience values before meeting the bound, as this might take some epochs
                    # But after that we update the trainer to use a smaller patience value again
                    self.trainer.early_stopping_callback.patience = int(self.trainer.early_stopping_callback.patience / 2)
                    print(f"Early stopping patience updated to: {self.trainer.early_stopping_callback.patience}")

                else:
                    assert len(existing_ckpts) == 1
                    ckpt_path = existing_ckpts[0]
                    cur_best_score = ckpt_path.split(self.metric_to_track + "=")[-1].split("-")[0]
                    cur_best_score = float(cur_best_score.strip(".ckpt"))

                    if "loss" in self.metric_to_track:
                        is_new_score_better = new_score <= cur_best_score
                    else:
                        is_new_score_better = new_score >= cur_best_score

                    if is_new_score_better:
                        save = True
                        print(f"Saving checkpoint based on meeting constrained optimization bound at epoch: {self.trainer.current_epoch} - {self.metric_to_track} improved from {cur_best_score:.5f} to {new_score:.5f}")
                        if os.path.isfile(ckpt_path):
                            os.remove(ckpt_path)
                        else:
                            raise Exception(f"Trying to delete file that does not exist: {ckpt_path}")

                        self._update_early_stopping_callback(new_score)

                if save:
                    avg_loss_ce = self.trainer.callback_metrics.get(f"val_avg_loss_ce").item()
                    avg_loss_annot = self.trainer.callback_metrics.get(f"val_avg_loss_annotation").item()
                    self.trainer.save_checkpoint(
                        os.path.join(
                            self.trainer.log_dir,
                            f"model-epoch={self.trainer.current_epoch:03d}-val_{self.metric_to_track}={new_score:.8f}-val_avg_loss_ce={avg_loss_ce:.8f}-val_avg_loss_annotation={avg_loss_annot:.8f}.ckpt",
                        )
                    )

    def test_step(self, batch, batch_idx):
        output = self._step(batch, is_train=False)
        return output

    def test_epoch_end(self, outputs):
        avg_loss = sum([o["loss"] for o in outputs]) / len(outputs)
        self.log("test_avg_loss", avg_loss.item())

        if "loss_ce" in outputs[0]:
            avg_loss_ce = sum([o["loss_ce"] for o in outputs]) / len(outputs)
            self.log("test_avg_loss_ce", avg_loss_ce.item())
        if "loss_annotation" in outputs[0]:
            avg_loss_annotation = sum([o["loss_annotation"] for o in outputs]) / len(outputs)
            self.log("test_avg_loss_annotation", avg_loss_annotation.item())
        if "loss_constrained" in outputs[0]:
            avg_loss_constrained = sum([o["loss_constrained"] for o in outputs]) / len(outputs)
            self.log("test_avg_loss_constrained", avg_loss_constrained.item())

        # Calculate F1-Score
        metrics = self.compute_metrics(outputs)
        self.log("test_f1_score", metrics['f1_macro'])
        self.log("test_acc_score", metrics['acc'])

    def predict_step(self, batch, batch_idx):
        output = self._step(batch, is_train=False, is_pred=True)
        return output

    def _step(self, batch, is_train):
        raise NotImplementedError

    def configure_optimizers(self):
        # Define optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, betas=(0.9, 0.98))

        # Define scheduler
        if self.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0.1*self.num_training_steps,
                num_training_steps=self.num_training_steps,
            )

            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        elif self.scheduler == "reduce_lr_on_plateau":
            scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_f1_score",
                "strict": True,
            }

        elif self.scheduler == "None":
            scheduler_config = None

        # Add constrained optimizer
        if self.constrained_optimization:
            constraint_optimizer = torch.optim.RMSprop([{"params": [self.alpha_cl], "lr": self.constrained_optimization_lr}])
            constraint_scheduler = get_linear_schedule_with_warmup(
                optimizer=constraint_optimizer,
                num_warmup_steps=0.1*self.num_training_steps,
                num_training_steps=self.num_training_steps,
            )
            constraint_scheduler_config = {
                "scheduler": constraint_scheduler,
                "interval": "step",
                "frequency": 1,
            }

        if scheduler_config:
            if self.constrained_optimization:
                return (
                    {"optimizer": optimizer, "lr_scheduler": scheduler_config},
                    {"optimizer": constraint_optimizer, "lr_scheduler": constraint_scheduler_config}, 
                )
            else:
                return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        else:
            if self.constrained_optimization:
                return ({"optimizer": optimizer}, {"optimizer": constraint_optimizer})
            else:
                return {"optimizer": optimizer}

    @property
    def num_training_steps(self) -> int:
        # Copied from: https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
        # Copied from: https://github.com/Zasder3/train-CLIP/issues/29
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches

        batches = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)
        #num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        num_devices = max(1, self.trainer.num_devices)
        #if self.trainer.tpu_cores:
        #    num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def freeze_embedding_model(self, approach):
        # Calculate number of layers of the model
        if "n_layers" in dir(self.model.config):
            nr_layers = self.model.config.n_layers
        else:
            nr_layers = self.model.config.num_hidden_layers

        # Freeze parameters of the model
        for param_name, param in self.model.bert.named_parameters():
            if approach == "all":
                param.requires_grad = False
            elif approach == "keep-top":
                param_name_fields = param_name.split(".")
                if param_name_fields[2].isdigit():
                    layer_nr = int(param_name_fields[2])
                    # Finetune only the top-layer
                    if layer_nr <= nr_layers - 1 - 1:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                else:
                    param.requires_grad = False

        # Print info
        print_sep()
        print("Embedding model layers being trained:")
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"ENCODER MODEL: Training layer {param_name}")
        print_sep()

    def compute_metrics(self, outputs):
        all_preds = torch.cat([o["preds"] for o in outputs]).tolist()
        all_golds = torch.cat([o["labels"] for o in outputs]).tolist()
        f1 = f1_score(y_pred=all_preds, y_true=all_golds, average="macro")
        acc = accuracy_score(y_pred=all_preds, y_true=all_golds)

        return {'f1_macro': f1, 'acc': acc}


class ModelModule(BaseModelModule):
    def __init__(self, args):
        super().__init__(args)

        self.lambda_annotation_loss = args.lambda_annotation_loss

        # Load embedding model
        #self.model = BFCS.from_pretrained(
        config = AutoConfig.from_pretrained(
            args.model_name,
            output_hidden_states=True,
            output_attentions=True,
            num_labels=2,
            classifier_dropout=0.0,
        )
        self.model = BigBirdForSequenceClassification.from_pretrained(args.model_name, config=config)

        if args.supervised_heads == "all":
            self.supervised_heads = list(range(self.model.config.num_attention_heads))
        else:
            self.supervised_heads = [int(x) for x in args.supervised_heads.split(",")]

        self.head_aggregation_method = args.head_aggregation_method

        guided_loss_map = {'mae': torch.nn.functional.l1_loss}
        self.guided_loss = guided_loss_map["mae"]

        self.attention_type = args.attention_type

        self.attr_scaling = args.attr_scaling

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
        parser.add_argument("--lambda_annotation_loss", type=float, default=0.0)
        parser.add_argument("--attention_type", type=str, default="top_layer")
        parser.add_argument("--supervised_heads", type=str, default="all")
        parser.add_argument("--head_aggregation_method", type=str, default="mean")
        parser.add_argument("--attr_scaling", type=int, default=1)
        return parent_parser

    @staticmethod
    def check_model_specific_args(args):
        # Assert supervised_heads is list of ints
        if args.supervised_heads != "all":
            heads = args.supervised_heads.split(",")
            for x in heads:
                try:
                    int(x)
                except:
                    raise ValueError("Invalid --supervised_heads, has non-int element")

        # Assert head_aggregation_methos is valid
        if args.head_aggregation_method not in ["none", "mean"]:
            raise ValueError(f"Invalid --head_aggregation_method {args.head_aggregation_method}")

        # Assert attention type is valid
        if args.attention_type not in ["top_layer", "rollout_top_layer", "all_layers"]:        
            raise ValueError(f"Invalid --attention_type {args.attention_type}")

        if args.attention_type in ["rollout_top_layer", "all_layers"] and args.head_aggregation_method == "none":
            raise ValueError(f"--attention_type {args.attention_type} does not support --head_aggregation_method {args.head_aggregation_method}")

        # Assert attr_scaling is larger than 1
        if args.attr_scaling < 1:
            raise ValueError(f"--attr_scaling {args.attr_scaling} should be >= 1")

    def forward(self, inputs_embeds, attention_mask, token_type_ids):

        # Encode input
        embedding_model_output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        return {
            "pred_logits": embedding_model_output["logits"],
            "attentions": embedding_model_output["attentions"],
            "hidden_states": embedding_model_output["hidden_states"],
        }

    def forward_captum(self, inputs, attention_mask, token_type_ids):
        pred = self.forward(inputs, attention_mask, token_type_ids)["pred_logits"]
        return pred

    # Override base model's _step
    def _step(self, batch, is_train, is_pred=False):
        forward_inputs = {
            "inputs_embeds" : self.model.bert.embeddings(batch["input_ids"]),
            "attention_mask" : batch["attention_mask"],
            "token_type_ids" : batch["token_type_ids"],
        }
        output = self.forward(**forward_inputs)

        # With BigBird hidden states and attentions might be output with an extra padding
        # Internally, the Transformers package only removes it for other data structures
        if output["attentions"][0].shape[-1] != batch["input_ids"].shape[-1]:
            pad_len = output["attentions"][0].shape[-1] - batch["input_ids"].shape[-1]
            assert pad_len > 1
            output["attentions"] = tuple((att[:, :, :-pad_len, :-pad_len] for att in output["attentions"]))
            output["hidden_states"] = tuple((hs[:, :-pad_len] for hs in output["hidden_states"]))

        labels = batch["labels"]

        # Calculate CE loss
        loss_ce = self.criterion(output["pred_logits"], target=labels)

        # Calculate annotation-based loss
        if "annotation_targets" not in batch.keys() and not self.use_annotation_loss_only:
            loss_annotation = torch.tensor(0.0, device=loss_ce.device)
        else:
            # Compute attributions
            attrs = self.compute_attributions(batch=batch, output=output)
            loss_annotation = self.compute_annotation_loss(attrs=attrs, batch=batch)

        # Build output dictionary
        if torch.isnan(loss_ce) or torch.isnan(loss_annotation):
            raise Exception("Either loss_ce or loss_annotation are NaN")
        if self.constrained_optimization:
            if self.constrained_optimization_loss == "classifier":
                main_loss = loss_annotation
                bounded_loss = loss_ce
            elif self.constrained_optimization_loss == "guided":
                main_loss = loss_ce
                bounded_loss = loss_annotation
            loss_constrained = self.alpha_cl * (bounded_loss - self.margin_cl)
            loss = main_loss + loss_constrained
        elif self.use_annotation_loss_only:
            loss = loss_annotation
        else:
            loss = loss_ce + self.lambda_annotation_loss * loss_annotation
        output_dict = {
            "loss": loss,
            "loss_ce": loss_ce.detach(),
            "loss_annotation": loss_annotation.detach(),
        }
        if self.constrained_optimization:
            output_dict["loss_constrained"] = loss_constrained.detach()
            #output_dict["loss_constrained_multiplier"] = self.loss_constraint._multiplier.detach().item()

        if "annotation_targets" in batch:
            output_dict["entropy_attributions"] = Categorical(attrs).entropy().mean().detach()

        # Get predictions
        if not is_train:
            preds = torch.argmax(output["pred_logits"], dim=1)
            output_dict["preds"] = preds
            output_dict["labels"] = labels
        
        #if is_pred:
            output_dict["pred_logits"] = output["pred_logits"]
            bs, num_heads, num_tokens, num_tokens = output["attentions"][-1].shape
            input_ids = []
            for iids, msk in zip(batch["input_ids"], batch["attention_mask"]):
                input_ids.append(iids[msk==1])

            # Collect the attention weights and attributions for the same batch across all layers. And then remove the masked tokens.
            # attention: list with batch size entries of dims [num_layers, num_tokens]
            attentions = []
            attributions = []
            for bs_ix in range(bs):
                num_non_masked_tokens = batch["attention_mask"][bs_ix].sum()
                # batch_attentions: [num_layers, num_heads, num_tokens, num_tokens]
                batch_attentions = torch.stack([l[bs_ix] for l in output["attentions"]])[:, :, :num_non_masked_tokens, :num_non_masked_tokens]
                # Keep only the attentions to the CLS token, and mean over heads
                attentions.append(batch_attentions[:, :, 0, :].mean(dim=1))
                if "annotation_targets" in batch:
                    # For attributions, we already have [batch_size, num_tokens]
                    attributions.append(attrs[bs_ix, :num_non_masked_tokens])

            output_dict["attentions"] = attentions
            output_dict["input_ids"] = input_ids
            if "annotation_targets" in batch and "annotation_keep_loss" in batch:
                annotation_targets = []
                for ann_tgt, msk in zip(batch["annotation_targets"], batch["attention_mask"]):
                    annotation_targets.append(ann_tgt[msk==1])

                output_dict["annotation_targets"] = annotation_targets
                output_dict["annotation_keep_loss"] = batch["annotation_keep_loss"]
                output_dict["attributions"] = attributions

        return output_dict

    def compute_attributions(self, batch, output):
        if self.head_aggregation_method == "none":
            if self.attention_type == "top_layer":
                # Get attention weights to be supervised
                # NOTE: This doesn't sum to one during training, because:
                # 1) HF applies dropout after softmax
                # https://github1s.com/huggingface/transformers/blob/HEAD/src/transformers/models/bert/modeling_bert.py#L358
                # 2) Dropout re-normalizes values by 1/(1-p)
                # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
                # NOTE: Since the call uses attention_mask, this already zero-outs attention to padding
                # attentions_to_supervise: [batch_size, num_heads, sequence_length, sequence_length]
                attentions_to_supervise = output["attentions"][-1]

                # Keep only attentions to CLS
                # attentions: [batch_size, num_heads, sequence_length]
                attrs = attentions_to_supervise[:, :, 0, :]

        elif self.head_aggregation_method == "mean":
                if self.attention_type == "top_layer":
                    attentions_to_supervise = output["attentions"][-1]
                    attentions_to_supervise = attentions_to_supervise[:, :, 0, :]
                    attrs = attentions_to_supervise.mean(dim=1)

                elif self.attention_type == "rollout_top_layer":
                    # In: [batch_size, num_layers, num_heads, num_tokens, num_tokens]
                    # Out: [batch_size, num_layers, num_tokens, num_tokens]
                    attention_rollout = self.compute_attention_rollout(torch.stack(output["attentions"], dim=1))
                    attrs = attention_rollout[:, -1, 0, :]

                elif self.attention_type == "all_layers":
                    attentions_to_supervise = torch.stack(output["attentions"], dim=1).mean(dim=(1, 2))
                    attrs = attentions_to_supervise[:, 0]

                # NOTE: This copies:
                # https://github.com/INK-USC/ER-Test/blob/HEAD/src/utils/losses.py#L16
                # https://github.com/INK-USC/ER-Test/blob/HEAD/src/utils/losses.py#L27-L28
                if self.attention_type in ['top_layer', 'rollout_top_layer']:
                    if self.attr_scaling != 1:
                        attrs = attrs * self.attr_scaling
                        attrs = torch.where(batch["attention_mask"] == 0, -1e6, attrs)
                        attrs = attrs.softmax(dim=-1)

        return attrs

    def compute_annotation_loss(self, attrs, batch):
        # Build annotation targets
        annotation_targets = batch["annotation_targets"].float()
        annotation_targets_denominator = torch.sum(annotation_targets, dim=1)
        # Some entries might not have highlighted words. We solve that issue in the denominator, but don't
        # calculate the loss for those cases, by zero-ing out those values in the sum
        annotation_loss_mask = (annotation_targets_denominator != 0).int()
        annotation_targets_denominator[annotation_targets_denominator == 0] = 1
        annotation_targets /= annotation_targets_denominator.unsqueeze(-1)

        # We want to exclude the cases that have no higlights for either premise or hypothesis
        annotation_loss_mask *= batch["annotation_keep_loss"]

        # Compute annotation loss
        if self.head_aggregation_method == "none":
            supervised_attention_heads = attrs[:, self.supervised_heads, :]
            loss_annotated_heads = self.guided_loss(
                supervised_attention_heads,
                annotation_targets.unsqueeze(-2).expand(-1, len(self.supervised_heads), -1),
                reduction='none'
            )
            loss_annotated_heads = torch.sum(loss_annotated_heads, dim=1)
            loss_annotated_heads *= annotation_loss_mask.unsqueeze(-1)
            loss_annotation = torch.sum(loss_annotated_heads)
            loss_annotation = torch.true_divide(loss_annotation, len(self.supervised_heads))

        elif self.head_aggregation_method == "mean":
            # This is equivalent to square then *= w/ .unsqueeze(-1) and then .sum()
            loss_annotated = self.guided_loss(attrs, annotation_targets, reduction='none')
            loss_annotated = torch.sum(loss_annotated, dim=1)
            loss_annotated *= annotation_loss_mask
            loss_annotation = torch.sum(loss_annotated)

        # Normalize annotation loss by the number of tokens in the batch
        # If no annotated examples in the batch, zero-out the loss
        if sum(annotation_loss_mask) == 0:
            loss_annotation *= 0
        else:
            loss_annotation /= (batch["attention_mask"] * annotation_loss_mask.unsqueeze(-1)).sum()

        return loss_annotation

    def compute_joint_attention(self, att_mat, add_residual=True):
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

        num_layers = aug_att_mat.shape[1]
        # [batch_size, num_layers, num_tokens, num_tokens]
        joint_attentions = aug_att_mat[:, 0].unsqueeze(1)
        for i in range(1, num_layers):
            new_joint_attentions = torch.bmm(aug_att_mat[:, i], joint_attentions[:, i-1])
            joint_attentions = torch.cat((joint_attentions, new_joint_attentions.unsqueeze(1)), dim=1)

        return joint_attentions

    def compute_attention_rollout(self, attentions, input_includes_heads=True):
        if input_includes_heads:
            # Compute mean over attention heads
            # attentions: [batch_size, num_layers, num_heads, num_tokens, num_tokens]
            # attentions_mean_heads: [batch_size, num_layers, num_tokens, num_tokens]
            attentions_mean_heads = attentions.mean(dim=2)
        else:
            attentions_mean_heads = attentions

        # Compute joint attentions
        joint_attentions = self.compute_joint_attention(attentions_mean_heads, add_residual=True)

        return joint_attentions


def main(args, model_module):
    pl.seed_everything(args.random_seed)

    if not (args.predict or args.eval):
        if args.checkpoint_path:
            print("\n\nResuming training from: ", args.checkpoint_path)
            print("WARNING: Args of the original model will not be used.\n\n")
            model = model_module.load_from_checkpoint(args.checkpoint_path, args=args)
        else:
            model = model_module(args)
        data = DataModule(args)
        data.prepare_data(stage="fit")
        trainer = Trainer(args, stage="fit").trainer

        # Train model
        trainer.fit(model, data)

        # Evaluate model with test data
        print_sep()
        print("TESTING:")
        data = DataModule(args)
        data.prepare_data(stage="eval")
        if args.constrained_optimization:
            ckpts = glob(os.path.join(trainer.log_dir, "*.ckpt"))
            if not ckpts:
                print("No checkpoint available - Bound was not met during training")
            else:
                assert len(ckpts) == 1
                ckpt_path = ckpts[0]
                trainer.test(ckpt_path=ckpt_path, dataloaders=data)
        else:
            trainer.test(ckpt_path="best", dataloaders=data)

    elif args.eval:
        data = DataModule(args)
        data.prepare_data("eval")

        trainer = Trainer(args, stage="eval").trainer

        # Find all checkpoint and sort by version
        if ".ckpt" in args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            ckpt_paths = [args.checkpoint_path]
        else:
            assert os.path.isdir(args.checkpoint_path)
            checkpoint_path = os.path.join(args.checkpoint_path, "version_*", "*.ckpt")
            ckpt_paths = [path for path in sorted(glob(checkpoint_path), key=lambda x: int(x.split("version_")[-1].split("/")[0]))]
            print(f"Found {len(ckpt_paths)} checkpoints")

        for ckpt_path in ckpt_paths:
            print(f"{'-'*10}\nPATH: {ckpt_path}")
            model = model_module.load_from_checkpoint(ckpt_path)
            trainer.test(model, dataloaders=data)

    elif args.predict:
        print("Checkpoint: ", args.checkpoint_path)
        model = model_module.load_from_checkpoint(args.checkpoint_path)
        data = DataModule(args)
        data.prepare_data(stage="predict")

        trainer = Trainer(args, stage="predict").trainer
        results = trainer.predict(model, dataloaders=data)

        # Concat all logits and get predictions
        all_preds = [r for batch_results in results for r in batch_results["preds"]]
        all_preds = torch.stack(all_preds)
        all_labels = [r for batch_results in results for r in batch_results["labels"]]
        all_labels = torch.stack(all_labels)
        all_logits = [r for batch_result in results for r in batch_result["pred_logits"]]
        all_logits = torch.stack(all_logits)
        all_attentions = [r for batch_result in results for r in batch_result["attentions"]]
        all_input_ids = [r for batch_result in results for r in batch_result["input_ids"]]

        # Get all tokens
        all_tokens = [data.tokenizer.convert_ids_to_tokens(x) for x in all_input_ids]

        assert len(all_preds) == len(all_labels)
        assert len(all_preds) == len(all_logits)
        assert len(all_preds) == len(all_attentions)
        assert len(all_preds) == len(all_input_ids)
        assert len(all_preds) == len(all_tokens)

        data_to_save = {}
        data_to_save["preds"] = all_preds
        data_to_save["labels"] = all_labels
        data_to_save["logits"] = all_logits
        data_to_save["attentions"] = all_attentions
        data_to_save["input_ids"] = all_input_ids
        data_to_save["tokens"] = all_tokens

        if "annotation_targets" in results[0] and "annotation_keep_loss" in results[0]:
            all_annotation_targets = [r for batch_result in results for r in batch_result["annotation_targets"]]
            all_annotation_keep_loss = [r for batch_results in results for r in batch_results["annotation_keep_loss"]]
            all_annotation_keep_loss = torch.stack(all_annotation_keep_loss)
            all_attributions = [r for batch_result in results for r in batch_result["attributions"]]

            assert len(all_preds) == len(all_annotation_targets)
            assert len(all_preds) == len(all_annotation_keep_loss)
            assert len(all_preds) == len(all_attributions)

            data_to_save["annotation_targets"] = all_annotation_targets
            data_to_save["annotation_keep_loss"] = all_annotation_keep_loss
            data_to_save["attributions"] = all_attributions

        # Save prediction results to joblib
        suffix = os.path.splitext(args.predict_data_filename)[0]
        save_path = os.path.join(os.path.split(args.checkpoint_path)[0], f"pred_{suffix}_data.joblib")
        print("Saving data to: ", save_path)
        joblib.dump(data_to_save, save_path, compress=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="tb_logger")
    parser.add_argument("--experiment_name", type=str, default="my_model")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)

    parser = DataModule.add_data_specific_args(parser)
    parser = BaseModelModule.add_base_model_specific_args(parser)
    parser = ModelModule.add_model_specific_args(parser)
    parser = Trainer.add_trainer_specific_args(parser)

    args = parser.parse_args()

    # Validate arguments
    ModelModule.check_model_specific_args(args)

    if args.checkpoint_path and not (args.predict or args.eval):
        print("\n\nWARNING: Starting training from existing checkpoint\n\n")

    if (args.predict or args.eval) and not args.checkpoint_path:
        raise Exception("Argument checkpoint_path is expected when using args.predict or args.eval")

    if not args.eval and not args.predict:
        print(f"ARGS:\n{args}\n")

    main(args, ModelModule)
