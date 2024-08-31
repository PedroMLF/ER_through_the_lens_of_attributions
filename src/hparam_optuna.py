import json
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from argparse import ArgumentParser
from glob import glob

import numpy as np
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.approach import BaseModelModule, ModelModule, IxGModule
from src.approach import DataModule
from src.ptl_trainer import Trainer as pl_trainer


def objective(trial, seed, metric_to_select, metric_to_early_stop, save_dir):

    print("\n\n")
    pl.seed_everything(seed)

    learning_rate = trial.suggest_float("learning_rate", 0, 99.0)
    batch_size = trial.suggest_int("train_batch_size", 1, 99)
    max_epochs = trial.suggest_int("max_epochs", 1, 99)
    lambda_annotation_loss = trial.suggest_float("lambda_annotation_loss", 0.0, 99.0)

    # Update args with the new values
    args_dict = vars(args)
    args_dict['train_batch_size'] = batch_size
    args_dict['eval_batch_size'] = batch_size
    args_dict['learning_rate'] = learning_rate
    args_dict['lambda_annotation_loss'] = lambda_annotation_loss

    if args.constrained_optimization:
        constrained_learning_rate = trial.suggest_float("constrained_learning_rate", 0, 99.0)
        args_dict["constrained_optimization_lr"] = constrained_learning_rate

    print(args)

    datamodule = DataModule(args)
    datamodule.prepare_data(stage="fit")

    # Get the right model class
    if args.attention_type == "IxG":
        model_module = IxGModule
    else:
        model_module = ModelModule
    model = model_module(args)

    # Define save_dir and experiment_name
    #save_dir = "optuna_hparam_search_folder"
    if args.constrained_optimization:
        experiment_name = f"bs-{batch_size}_lr-{learning_rate}_clr-{constrained_learning_rate}_ep-{max_epochs}_ld-{str(lambda_annotation_loss).replace('.', 'pt')}"
    else:
        experiment_name = f"bs-{batch_size}_lr-{learning_rate}_ep-{max_epochs}_ld-{str(lambda_annotation_loss).replace('.', 'pt')}"

    # Define logger
    logger = TensorBoardLogger(save_dir=save_dir, name=experiment_name)

    # Define early stop callback
    early_stop_callback = EarlyStopping(
        monitor=f"val_{args.metric_to_early_stop}",
        min_delta=0.00,
        patience=4 if not args.constrained_optimization else 10,
        verbose=True,
        mode="min" if "loss" in args.metric_to_early_stop else "max",
    )

    # Define checkpoint callback
    filename_str = "model-{epoch:03d}-{val_acc_score:.6f}-{val_f1_score:.6f}-{val_avg_loss_ce:.8f}-{val_avg_loss:.8f}"

    # Define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=f"val_{args.metric_to_early_stop}",
        dirpath=os.path.join(save_dir, logger.name, f"version_{logger.version}"),
        filename=filename_str,
        save_top_k=1,
        save_last=False,
        mode="min" if "loss" in args.metric_to_early_stop else "max",
        save_weights_only=True,
    )

    # We do manual checkpointing when doing constrained optimization
    if args.constrained_optimization:
        callbacks = [early_stop_callback]
    else:
        callbacks = [early_stop_callback, checkpoint_callback]

    trainer = pl_trainer(
        accelerator="auto",
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=False,
        gpus=1,
        logger=logger,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        profiler=None,
        enable_checkpointing=False if args.constrained_optimization else True,
        inference_mode=False if args.attention_type == "IxG" else True,
    )

    trainer.fit(model, datamodule=datamodule)

    if args.constrained_optimization:
        model_paths = glob(os.path.join(trainer.log_dir, "*.ckpt"))
        if len(model_paths) > 1:
            raise Exception(f"At most one path expected in folder {trainer.log_dir}")
        elif not model_paths:
            print(f"Bound was not met for experiment {experiment_name}")
            score = 1e9 if "loss" in args.metric_to_select else 0
            return score
        else:
            best_model_path = model_paths[0]
    else:
        best_model_path = trainer.checkpoint_callback.best_model_path

    print("Best Checkpoint Path: ", best_model_path)

    # Read metric from checkpoint path and assert it matches early stop
    pattern_metric_to_early_stop = r"{}=([\d.]+)".format(args.metric_to_early_stop)
    match_metric_to_early_stop = re.search(pattern_metric_to_early_stop, best_model_path)
    score = match_metric_to_early_stop.group(1)
    if score[-1] == ".": score = score[:-1]
    assert np.isclose(float(score), trainer.early_stopping_callback.best_score.item())

    # Read metric to select from checkpoint path and return it
    pattern_metric_to_select = r"{}=([\d.]+)".format(args.metric_to_select)
    match_metric_to_select = re.search(pattern_metric_to_select, best_model_path)
    score_to_select = match_metric_to_select.group(1)
    if score_to_select[-1] == ".": score_to_select = score_to_select[:-1]
    score_to_select = float(score_to_select)

    return score_to_select

def objective_wrapper(trial, seeds, prune, metric_to_select, metric_to_early_stop, save_dir):

    print("\n-------------------")
    print("---- New Run -----")
    print("------------------")

    all_results = []
    run_is_pruned = False

    seeds = seeds.split(",")

    for seed_ix, seed in enumerate(seeds):
        seed = int(seed)
        result = objective(
            trial,
            seed=seed,
            metric_to_select=metric_to_select,
            metric_to_early_stop=metric_to_early_stop,
            save_dir=save_dir,
        )
        all_results.append(result)

        # Manually prune, since we cannot run multiple seeds and use the existing methods
        if prune and len(trial.study.trials) > 1 and seed_ix > 1:
            best_mean = np.mean(trial.study.best_trial.user_attrs['individual_seed_results'])
            best_std = np.std(trial.study.best_trial.user_attrs['individual_seed_results'])
            if (np.mean(all_results) + np.std(all_results)) < (best_mean - best_std):
                print("-- Trial is being pruned")
                run_is_pruned = True
                break

    if run_is_pruned:
        all_results.extend([-1] * (len(seeds) - len(all_results)))

    trial.set_user_attr("individual_seed_results", all_results)

    print(f"\nMean: {np.mean(all_results):.5f} -- STD: {np.std(all_results):.5f} -- All: {all_results}\n")

    time.sleep(5)

    return sum(all_results) / len(all_results)

def main(args):
    search_space = json.load(open(args.search_space_path))['search_space']
    sampler = optuna.samplers.GridSampler(search_space)

    study = optuna.create_study(
        study_name="optim test",
        direction="minimize" if "loss" in args.metric_to_select else "maximize",
        sampler=sampler,
    )

    study.optimize(
        lambda trial: objective_wrapper(
            trial,
            seeds=args.seeds,
            prune=args.pruning,
            metric_to_select=args.metric_to_select,
            metric_to_early_stop=args.metric_to_early_stop,
            save_dir=args.save_dir,
        ),
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--search_space_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/bigbird-roberta-base")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument(
        "--metric_to_select",
        type=str,
        help="Metric used to select the set of best performing hparams."
    )
    parser.add_argument(
        "--metric_to_early_stop",
        type=str,
        help="Metric used to do early stopping / checkpoint selection during training (non-constrained)."
    )
    parser.add_argument(
        "--metric_to_track",
        type=str,
        default=None,
        help="Metric used by constrained optimization to select the best checkpoint provided that bound is met."
    )
    parser.add_argument("--pruning", action="store_true")

    parser = DataModule.add_data_specific_args(parser)
    parser = BaseModelModule.add_base_model_specific_args(parser)
    parser = ModelModule.add_model_specific_args(parser)

    args = parser.parse_args()

    # Validate arguments
    ModelModule.check_model_specific_args(args)

    if args.metric_to_track:
        assert args.constrained_optimization

    print(f"DEFAULT ARGS:\n{args}\n")

    main(args)
