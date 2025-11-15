"""
Sherpa-driven hyperparameter optimisation entrypoint for the HPST Lightning trainer.

This script mirrors the behaviour of ``scripts/train.py`` while delegating the actual
search to Sherpa.  Each trial instantiates a brand-new ``HeterogenousPointSetTrainer``
with overrides sampled from the parameter space defined around
``config/hpst/hpst_tune_nova.json``.  Results (including Sherpa's ``results.csv``)
are written under the requested ``--output_dir`` (defaults to ``./sherpa/hpst``).
"""

from __future__ import annotations

import argparse
import copy
import datetime
import importlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import pandas as pd
import torch


def _ensure_sherpa_without_gpy() -> None:
    """Install lightweight stubs so Sherpa can import without the GPy/GPyOpt stack."""
    def _maybe_stub(module_name: str, builder):
        if module_name in sys.modules:
            return
        try:
            importlib.import_module(module_name)
        except Exception:
            sys.modules[module_name] = builder()

    def _build_gpy():
        mod = types.ModuleType("GPy")

        class _Unavailable:
            def __getattr__(self, _):
                raise ImportError(
                    "GPy functionality is disabled. Use --algorithm random (default)."
                )
        mod.kern = _Unavailable()
        mod.models = _Unavailable()
        return mod

    def _build_gpyopt():
        mod = types.ModuleType("GPyOpt")

        class _BayesOptDisabled:
            def __init__(self, *_, **__):
                raise ImportError(
                    "GPyOpt is unavailable; only RandomSearch is supported."
                )
        mod.methods = types.SimpleNamespace(BayesianOptimization=_BayesOptDisabled)
        return mod

    _maybe_stub("GPy", _build_gpy)
    _maybe_stub("GPyOpt", _build_gpyopt)


_ensure_sherpa_without_gpy()

from sherpa import Choice, Continuous, Discrete, Parameter, Study
from sherpa.algorithms import RandomSearch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Restore deprecated pandas.DataFrame.append required by Sherpa.
if not hasattr(pd.DataFrame, "append"):
    def _df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        return pd.concat([self, other], ignore_index=ignore_index, sort=sort)
    pd.DataFrame.append = _df_append  # type: ignore[attr-defined]

from hpst.utils.options import Options  # noqa: E402
from hpst.trainers.heterogenous_point_set_trainer import HeterogenousPointSetTrainer  # noqa: E402


class BestMetricTracker(Callback):
    """Tracks the best value reached for a monitored metric during training."""

    def __init__(self, metric_name: str = "val_accuracy") -> None:
        super().__init__()
        self.metric_name = metric_name
        self.best = float("-inf")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:  # pragma: no cover - Lightning hook
        metric = trainer.callback_metrics.get(self.metric_name)
        if metric is None:
            return
        if isinstance(metric, torch.Tensor):
            metric_value = metric.detach().float().item()
        else:
            metric_value = float(metric)
        if metric_value > self.best:
            self.best = metric_value


class TrialProgressPrinter(Callback):
    """Lightweight console progress output for each trial."""

    def __init__(self, total_epochs: int, description: str) -> None:
        super().__init__()
        self.total_epochs = max(1, int(total_epochs))
        self.description = description

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if isinstance(value, torch.Tensor):
            return value.detach().float().item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _metric(
        self, trainer: pl.Trainer, *names: str
    ) -> Optional[float]:  # pragma: no cover - simple helper
        metrics = trainer.callback_metrics
        for name in names:
            if name in metrics:
                return self._to_float(metrics[name])
        return None

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:  # pragma: no cover - Lightning hook
        print(f"{self.description} starting ({self.total_epochs} epochs planned).", flush=True)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:  # pragma: no cover - Lightning hook
        epoch_idx = int(trainer.current_epoch or 0) + 1
        train_loss = self._metric(trainer, "train_loss", "total_train_loss")
        val_loss = self._metric(trainer, "val_loss")
        val_acc = self._metric(trainer, "val_accuracy")
        parts = [f"Epoch {epoch_idx}/{self.total_epochs}"]
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if val_acc is not None:
            parts.append(f"val_acc={val_acc:.4f}")
        print(f"{self.description} | " + " ".join(parts), flush=True)

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:  # pragma: no cover - Lightning hook
        completed = trainer.current_epoch + 1 if trainer.current_epoch is not None else self.total_epochs
        print(f"{self.description} finished after {completed} epochs.", flush=True)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sherpa hyperparameter optimisation for the HPST network."
    )
    parser.add_argument(
        "--options_file",
        type=str,
        default="config/hpst/hpst_tune_nova.json",
        help="JSON file providing the baseline HPST options.",
    )
    parser.add_argument(
        "--training_file",
        type=str,
        default=None,
        help="Optional override for the training dataset path.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=64,
        help="Maximum number of Sherpa trials to execute.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="random",
        choices=["random"],
        help="Sherpa search algorithm (GPyOpt disabled; random search only).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("sherpa", "hpst"),
        help="Directory where Sherpa should place results.csv and metadata.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Validation patience (epochs) for early stopping callbacks.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=16,
        help="Optional hard cap for epochs per trial. Overrides config if set.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="GPUs to request per trial (defaults to value in options file).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of dataloader workers to use (defaults to options file).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6657,
        help="Base RNG seed. Individual trials add their index to this seed.",
    )
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=20000,
        help="Optional cap on the number of training samples per Sherpa trial. Max: 2M. Set to None for all.",
    )
    parser.add_argument(
        "--val-max-samples",
        type=int,
        default=6000,
        help="Optional cap on the number of validation samples per Sherpa trial. Max: 346k",
    )
    parser.add_argument(
        "--use_wandb", 
        action="store_true", 
        help="Enable WandB logging."
    )
    parser.add_argument(
        "--wandb_project", t
        ype=str, 
        default="HPST", 
        help="WandB project name."
    )
    parser.add_argument(
        "--logdir", 
        type=str, 
        default="runs", 
        help="Base log directory."
    )

    return parser.parse_args()


def _load_base_options(args: argparse.Namespace) -> Options:
    options = Options(training_file=args.training_file or "")
    with open(args.options_file, "r", encoding="utf-8") as file_obj:
        options.update_options(json.load(file_obj))

    if args.training_file:
        options.training_file = args.training_file
    if args.gpus is not None:
        options.num_gpu = max(0, args.gpus)
    if args.num_workers is not None:
        options.num_dataloader_workers = max(0, args.num_workers)
    if args.max_epochs is not None:
        options.epochs = max(1, args.max_epochs)
    if args.train_max_samples is not None:
        options.train_max_samples = max(0, args.train_max_samples)
    if args.val_max_samples is not None:
        options.val_max_samples = max(0, args.val_max_samples)

    options.verbose_output = False
    return options


def _parameter_space() -> List[Parameter]:
    """Fixed Sherpa parameter space tailored to hpst_tune_nova.json."""
    return [
        Continuous("learning_rate", [1e-5, 1e-2], scale="log"),
        Continuous("l2_penalty", [1e-4, 1e-1], scale="log"),
        Choice("batch_size", [256, 384, 512]),
        Continuous("gradient_clip", [16.0, 80.0]),
        Discrete("learning_rate_warmup_epochs", [2, 4, 6, 8]),
        Discrete("learning_rate_cycles", [1, 2, 3]),
        Continuous("loss_gamma", [1.0, 4.0]),
        Continuous("loss_beta", [0.999, 0.9999999]),
        Choice("epochs", [16]),
    ]



def _prepare_wandb_logger(args, trial_index, options):
    """Create a WandB logger consistent with train.py logging style."""
    if not args.use_wandb:
        return None

    # Create a unique readable run name like trial_003
    run_name = f"trial_{trial_index:03d}"

    # Mimic train.py structure: logs under {logdir}/hpst/<timestamp>/
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    run_id = f"{timestamp}_t{trial_index}"
    base_dir = Path(args.logdir) / "hpst" / run_id

    logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        id=f"hpst_trial_{trial_index}",
        save_dir=str(base_dir.parent),
        log_model=False,
        resume="allow",
    )

    # Match train.py → update_config(logger, config_dict)
    logger.experiment.config.update(vars(options), allow_val_change=True)

    return logger



def _apply_trial_parameters(options: Options, params: Dict[str, Any]) -> None:
    for key, value in params.items():
        if not hasattr(options, key):
            continue
        current = getattr(options, key)
        if isinstance(current, bool):
            setattr(options, key, bool(value))
        elif isinstance(current, int) and not isinstance(value, bool):
            setattr(options, key, int(round(value)))
        elif isinstance(current, float):
            setattr(options, key, float(value))
        else:
            setattr(options, key, value)

    options.batch_size = max(1, int(options.batch_size))
    options.epochs = max(1, int(options.epochs))
    options.learning_rate = max(1e-6, float(options.learning_rate))
    options.l2_penalty = max(0.0, float(options.l2_penalty))
    options.gradient_clip = max(0.0, float(options.gradient_clip))


def _resolve_devices(requested_gpus: int) -> Tuple[str, int]:
    available = torch.cuda.device_count()
    if requested_gpus > 0 and available > 0:
        return "gpu", min(requested_gpus, available)
    return "cpu", 1


def _build_trainer(
    options: Options, patience: int, trial_desc: str, logger: Optional[WandbLogger]
) -> Tuple[pl.Trainer, BestMetricTracker]:
    accelerator, devices = _resolve_devices(options.num_gpu)
    options.num_gpu = devices if accelerator == "gpu" else 0

    tracker = BestMetricTracker("val_accuracy")
    callbacks: List[Callback] = [tracker, TrialProgressPrinter(options.epochs, trial_desc)]
    if patience and patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                mode="max",
                check_on_train_epoch_end=False,
                verbose=False,
            )
        )

    trainer = pl.Trainer(
        max_epochs=options.epochs,
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        gradient_clip_val=options.gradient_clip,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
    )
    return trainer, tracker


def _create_algorithm(name: str, max_trials: int):
    """Return the configured Sherpa algorithm (currently RandomSearch only)."""
    if name.lower() != "random":
        raise ValueError("Only 'random' algorithm is supported now that GPyOpt is disabled.")
    return RandomSearch(max_num_trials=max_trials)


def _run_single_trial(
    trial_index: int,
    total_trials: int,
    trial,
    args: argparse.Namespace,
    base_options: Options,
) -> Dict[str, Any]:
    trial_desc = f"Trial {trial_index}/{total_trials}"
    print(f"\n=== {trial_desc} | params: {trial.parameters} ===")
    pl.seed_everything(args.seed + trial_index, workers=True)

    trial_options: Options = copy.deepcopy(base_options)
    _apply_trial_parameters(trial_options, trial.parameters)

    trial_logger = _prepare_wandb_logger(args, trial_index, trial_options)
    trainer, tracker = _build_trainer(trial_options, args.patience, trial_desc, trial_logger)
    model = HeterogenousPointSetTrainer(trial_options)

    try:
        trainer.fit(model)
    finally:
        if trial_logger is not None:
            trial_logger.experiment.finish()

    best_val = tracker.best if tracker.best != float("-inf") else 0.0
    epochs_completed = (
        int(trainer.current_epoch) + 1 if trainer.current_epoch is not None else trial_options.epochs
    )

    metrics = {
        "val_accuracy": float(best_val),
        "epochs": int(epochs_completed),
        "batch_size": int(trial_options.batch_size),
        "learning_rate": float(trial_options.learning_rate),
        "l2_penalty": float(trial_options.l2_penalty),
        "gradient_clip": float(trial_options.gradient_clip),
        "learning_rate_warmup_epochs": float(trial_options.learning_rate_warmup_epochs),
        "learning_rate_cycles": int(trial_options.learning_rate_cycles),
        "loss_gamma": float(trial_options.loss_gamma),
        "loss_beta": float(trial_options.loss_beta),
    }

    # Free GPU memory between trials.
    del model
    torch.cuda.empty_cache()

    print(
        f"Trial {trial_index} finished | epochs: {epochs_completed} "
        f"| best val_accuracy: {metrics['val_accuracy']:.4f}"
    )
    return metrics


def run() -> None:
    args = _parse_arguments()
    pl.seed_everything(args.seed, workers=True)

    base_options = _load_base_options(args)
    parameter_space = _parameter_space()
    os.makedirs(args.output_dir, exist_ok=True)
    algorithm = _create_algorithm(args.algorithm, args.trials)

    study = Study(
        parameters=parameter_space,
        algorithm=algorithm,
        lower_is_better=False,
        disable_dashboard=True,
        output_dir=args.output_dir,
    )

    completed = 0
    for trial in study:
        completed += 1
        metrics = _run_single_trial(completed, args.trials, trial, args, base_options)
        study.add_observation(
            trial=trial,
            iteration=metrics["epochs"],
            objective=metrics["val_accuracy"],
            context=metrics,
        )
        study.finalize(trial)
        if completed >= args.trials:
            break

    study.save()
    best_result = study.get_best_result()
    summary_path = os.path.join(args.output_dir, "sherpa_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(best_result, file_obj, indent=2)

    print("\n=== Sherpa optimisation complete ===")
    print(f"Results directory : {args.output_dir}")
    print(f"Best result       : {best_result}")
    print(f"Summary JSON saved: {summary_path}")


if __name__ == "__main__":
    run()
