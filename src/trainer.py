import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.ptl_trainer import Trainer as pl_trainer


class Trainer:
    def __init__(self, args, stage, attn_type):

        if stage == "fit":
            # Define logger
            logger = TensorBoardLogger(save_dir=args.output_dir, name=args.experiment_name)

            # Define early stopping callback
            early_stop_callback = EarlyStopping(
                monitor=f"val_{args.metric_to_track}",
                min_delta=1e-5,
                patience=args.early_stopping_patience,
                verbose=True,
                mode="min" if "loss" in args.metric_to_track else "max",
            )

            if args.metric_to_track == "f1_score":
                filename_str = "model-{epoch:03d}-{val_f1_score:.5f}-{val_avg_loss:.5f}"
            else:
                filename_str = "model-{epoch:03d}-{val_acc_score:.5f}-{val_f1_score:.5f}-{val_avg_loss:.5f}"

            # Define model checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                monitor=f"val_{args.metric_to_track}",
                dirpath=os.path.join(args.output_dir, logger.name, f"version_{logger.version}"),
                filename=filename_str,
                save_top_k=1,
                save_last=False,
                mode="min" if "loss" in args.metric_to_track else "max",
                save_weights_only=True,
            )

            # Define learning rate monitor callback
            lr_monitor_callback = LearningRateMonitor(logging_interval="step")

            # We manually save checkpoints when doing constrained optimization
            callbacks = [checkpoint_callback, early_stop_callback, lr_monitor_callback]
            if args.constrained_optimization:
                callbacks.pop(0)

            # Set pytorch lightning trainer
            trainer_kwargs = {
                'accumulate_grad_batches': args.accumulate_grad_batches,
                'callbacks': callbacks,
                'deterministic': True,
                'enable_progress_bar': args.enable_progress_bar,
                'gpus': args.gpus,
                'log_every_n_steps': args.log_every_n_steps,
                'logger': logger,
                'max_epochs': args.max_epochs,
                'num_sanity_val_steps': 0,
                'overfit_batches': args.overfit_batches,
                'precision': 32,
                'profiler': None,
                'val_check_interval': args.val_check_interval,
                'enable_checkpointing': False if args.constrained_optimization else True,
                'inference_mode': False if attn_type == "IxG" else True,
            }
            if not args.constrained_optimization:
                trainer_kwargs['gradient_clip_val'] = args.gradient_clip_val

            self.trainer = pl_trainer(**trainer_kwargs)

        else:
            # Set minimal pytorch lightning trainer for eval/predict
            self.trainer = pl_trainer(
                deterministic=True,
                enable_progress_bar=args.enable_progress_bar,
                gpus=args.gpus,
                logger=None,
                profiler=None,
                precision=32,
                inference_mode=False if attn_type == "IxG" else True,
            )

    @staticmethod
    def add_trainer_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("trainer")

        parser.add_argument("--accumulate_grad_batches", type=int, default=1)
        parser.add_argument("--early_stopping_patience", type=int, default=40)
        parser.add_argument("--enable_progress_bar", action="store_true")
        parser.add_argument("--gpus", type=int, default=1)
        parser.add_argument("--gradient_clip_val", type=float, default=0)
        parser.add_argument("--log_every_n_steps", type=int, default=100)
        parser.add_argument("--max_epochs", type=int, default=100)
        parser.add_argument("--overfit_batches", type=float, default=0.0)
        parser.add_argument("--val_check_interval", type=float, default=0.25)
        parser.add_argument("--metric_to_track", type=str, default="f1")

        return parent_parser
