"""
Sherpa-based hyperparameter optimisation entrypoint.

This script keeps training (`model_train.py`) focused on a single run while
providing a separate CLI for hyperparameter studies. Usage example:

    python model_optim.py --model resnet18 --trials 24
"""

import argparse
import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sherpa import Choice, Continuous, Discrete, Parameter, Study, Trial
from sherpa.algorithms import Algorithm, GPyOpt, RandomSearch

from train import build_model
from utils import classifier_dataloader_cropped
import torch.optim.lr_scheduler as lr_scheduler

# Restore deprecated pandas.DataFrame.append required by Sherpa.
if not hasattr(pd.DataFrame, "append"):
    def _df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        return pd.concat([self, other], ignore_index=ignore_index, sort=sort)
    pd.DataFrame.append = _df_append  # type: ignore[attr-defined]

# Optional caps for the number of samples used during optimisation trials.
# Set to an integer (e.g., 1024) to subsample, or leave as None to use all data.
OPTIM_TRAIN_MAX_SAMPLES: Optional[int] = 8000
OPTIM_VAL_MAX_SAMPLES: Optional[int] = 4000

def _make_serializable(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - fall back below
            pass
    try:
        return float(value)
    except Exception:  # pragma: no cover - final fallback
        return str(value)


def _clean_best_result_dict(d: Dict[Any, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for k, v in d.items():
        # Drop purely numeric column names like 0 introduced by pandas quirks
        if isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
            continue
        cleaned[str(k)] = _make_serializable(v)
    # Ensure explicit val_acc alongside Objective
    if "val_acc" not in cleaned and "Objective" in cleaned:
        cleaned["val_acc"] = _make_serializable(cleaned["Objective"])
    return cleaned


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sherpa hyperparameter optimisation for event classification models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet",
        choices=["mobilenet", "resnet18", "resnet34", "resnet50"],
        help="Model architecture to optimise.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=32,
        help="Maximum number of Sherpa trials.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="bayes",
        choices=["bayes", "random"],
        help="Search algorithm: Bayesian optimisation (bayes) or random search.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store Sherpa results (defaults to ./<model>/sherpa).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience for validation loss within a trial.",
    )
    return parser.parse_args()


def _create_algorithm(name: str, max_trials: int) -> Algorithm:
    name = name.lower()
    if name == "bayes":
        return GPyOpt(max_num_trials=max_trials)
    if name == "random":
        return RandomSearch(max_num_trials=max_trials)
    raise ValueError(f"Unsupported algorithm: {name}")


def _parameter_space(model_name: str) -> Iterable[Parameter]:
    if model_name == "mobilenet":
        return [
            Continuous("lr", [1e-6, 1e-3], scale="log"),
            Choice("batch_size", [16]),
            # Choice("width_mult", [0.75, 1.0, 1.25]),
            # Continuous("label_smoothing", [0.0, 0.0]),
            Continuous("weight_decay", [1e-6, 1e-3], scale="log"),
            Choice("optimizer", ["adamw"]),
            Choice("scheduler", ["plateau"]),
            # Continuous("plateau_factor", [0.2, 0.8]),
            # Discrete("plateau_patience", [3, 5, 8]),
            # Continuous("min_lr", [1e-7, 1e-4], scale="log"),
            Choice("max_epochs", [10]),
        ]
    return [
        Continuous("lr", [1e-6, 1e-3], scale="log"),
        Choice("optimizer", ["sgd"]),
        Continuous("momentum", [0.2, 0.8]),
        Choice("batch_size", [8]),
        Continuous("weight_decay", [1e-7, 1e-4], scale="log"),
        # Choice("scheduler", ["step"]),
        # Discrete("step_size", [20]),
        # Choice("step_gamma", [0.5]),
        Choice("max_epochs", [10]),
    ]


def _limit_dataset(dataset, max_samples: Optional[int]):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    indices = torch.randperm(len(dataset))[: max_samples].tolist()
    return Subset(dataset, indices)


def _build_scheduler(
    model_name: str, optimizer: optim.Optimizer, config: Dict[str, float]
) -> Tuple[Optional[lr_scheduler._LRScheduler], Optional[str]]:
    scheduler_choice = config.get("scheduler")
    if not scheduler_choice:
        return None, None

    scheduler_choice = scheduler_choice.lower()
    if scheduler_choice == "plateau":
        factor = float(config.get("plateau_factor", 0.5))
        patience = int(config.get("plateau_patience", 5))
        min_lr = float(config.get("min_lr", 1e-6))
        return (
            lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=max(1e-6, min(factor, 0.99)),
                patience=max(1, patience),
                min_lr=min_lr,
            ),
            "val_loss",
        )

    if scheduler_choice == "step":
        step_size = int(config.get("step_size", 10))
        gamma = float(config.get("step_gamma", 0.5))
        return (
            lr_scheduler.StepLR(optimizer, step_size=max(1, step_size), gamma=gamma),
            "epoch",
        )

    return None, None


def _build_training_components(
    model_name: str, config: Dict[str, float], device: torch.device
) -> Tuple[nn.Module, nn.Module, optim.Optimizer, Optional[lr_scheduler._LRScheduler], Optional[str]]:
    model_kwargs: Dict[str, float] = {}
    if model_name == "mobilenet":
        model_kwargs["width_mult"] = config.get("width_mult", 1.0)
    model = build_model(model_name, **model_kwargs).to(device)

    if model_name == "mobilenet":
        smoothing = float(config.get("label_smoothing", 0.0))
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        weight_decay = float(config.get("weight_decay", 0.0))
        optimizer_choice = config.get("optimizer", "adam").lower()
        if optimizer_choice == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=weight_decay)
    else:
        criterion = nn.CrossEntropyLoss()
        weight_decay = float(config.get("weight_decay", 2e-6))
        optimizer_choice = config.get("optimizer", "adam").lower()
        if optimizer_choice == "sgd":
            momentum = float(config.get("momentum", 0.9))
            optimizer = optim.SGD(
                model.parameters(),
                lr=float(config["lr"]),
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            beta1 = float(config.get("beta1", 0.9))
            beta2 = float(config.get("beta2", 0.999))
            optimizer = optim.Adam(
                model.parameters(),
                lr=float(config["lr"]),
                betas=(beta1, beta2),
                weight_decay=weight_decay,
            )
    scheduler, scheduler_mode = _build_scheduler(model_name, optimizer, config)
    return model, criterion, optimizer, scheduler, scheduler_mode


def _prepare_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_loader_full, _ = classifier_dataloader_cropped(batch_size, shuffle=True)
    _, val_loader_full = classifier_dataloader_cropped(batch_size, shuffle=False)

    train_dataset = _limit_dataset(train_loader_full.dataset, OPTIM_TRAIN_MAX_SAMPLES)
    val_dataset = _limit_dataset(val_loader_full.dataset, OPTIM_VAL_MAX_SAMPLES)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def _tensorize_labels(labels, device: torch.device) -> torch.Tensor:
    return torch.tensor([label.to(device) for label in labels], dtype=torch.int64, device=device)


def _run_single_trial(
    study: Study,
    trial: Trial,
    model_name: str,
    device: torch.device,
    patience: int,
    trial_index: int,
    total_trials: int,
) -> Dict[str, float]:
    config = trial.parameters
    batch_size = int(config["batch_size"])
    max_epochs = int(config["max_epochs"])

    train_loader, val_loader = _prepare_dataloaders(batch_size)
    print(
        f"Trial {trial_index}/{total_trials} using "
        f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples."
    )
    model, criterion, optimizer, scheduler, scheduler_mode = _build_training_components(
        model_name, config, device
    )

    best_metrics = {"accuracy": 0.0, "val_loss": float("inf")}
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        train_desc = (
            f"Trial {trial_index}/{total_trials} | Epoch {epoch + 1}/{max_epochs} [train]"
        )
        train_bar = tqdm(train_loader, desc=train_desc, leave=False, dynamic_ncols=True)

        for image0s, image1s, labels in train_bar:
            image0s = image0s.float().to(device)
            image1s = image1s.float().to(device)
            labels = _tensorize_labels(labels, device)

            optimizer.zero_grad()
            outputs = model(image0s, image1s)
            loss = criterion(outputs, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * image0s.size(0)
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

            current_loss = train_loss_sum / max(train_total, 1)
            current_acc = train_correct / max(train_total, 1)
            train_bar.set_postfix(loss=f"{current_loss:.4f}", accuracy=f"{current_acc:.4f}")

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_desc = (
                f"Trial {trial_index}/{total_trials} | Epoch {epoch + 1}/{max_epochs} [val]"
            )
            val_bar = tqdm(val_loader, desc=val_desc, leave=False, dynamic_ncols=True)
            for image0s, image1s, labels in val_bar:
                image0s = image0s.float().to(device)
                image1s = image1s.float().to(device)
                labels = _tensorize_labels(labels, device)

                logits = model(image0s, image1s)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * image0s.size(0)
                _, preds = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                current_val_loss = val_loss_sum / max(val_total, 1)
                current_val_acc = val_correct / max(val_total, 1)
                val_bar.set_postfix(loss=f"{current_val_loss:.4f}", accuracy=f"{current_val_acc:.4f}")

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_acc = val_correct / max(val_total, 1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Trial {trial_index}/{total_trials} | Epoch {epoch + 1}/{max_epochs} "
            f"- train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} lr: {current_lr:.2e}"
        )

        if scheduler is not None:
            if scheduler_mode == "val_loss":
                scheduler.step(val_loss)
            elif scheduler_mode == "val_acc":
                scheduler.step(val_acc)
            else:
                scheduler.step()

        study.add_observation(
            trial=trial,
            iteration=epoch,
            objective=val_acc,
            context={
                "val_loss": val_loss,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": current_lr,
            },
        )

        if val_acc > best_metrics["accuracy"]:
            best_metrics["accuracy"] = val_acc
        if val_loss < best_metrics["val_loss"]:
            best_metrics["val_loss"] = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience or study.should_trial_stop(trial):
            break

    study.finalize(trial)
    torch.cuda.empty_cache()
    return best_metrics


def optimise_with_sherpa(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Sherpa optimisation for {args.model} on {device}.")
    print(f"Max trials: {args.trials} | Algorithm: {args.algorithm}")

    output_dir = args.output_dir or os.path.join(".", args.model, "sherpa")
    os.makedirs(output_dir, exist_ok=True)

    parameters = _parameter_space(args.model)
    algorithm = _create_algorithm(args.algorithm, args.trials)

    study = Study(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=False,
        disable_dashboard=True,
        output_dir=output_dir,
    )

    completed_trials = 0
    for trial in study:
        completed_trials += 1
        print(f"\n=== Trial {completed_trials} / {args.trials} | params: {trial.parameters} ===")
        best_metrics = _run_single_trial(
            study,
            trial,
            args.model,
            device,
            args.patience,
            trial_index=completed_trials,
            total_trials=args.trials,
        )
        print(
            f"Trial {completed_trials} finished | Best accuracy: {best_metrics['accuracy']:.4f} "
            f"| Best val loss: {best_metrics['val_loss']:.4f}"
        )
        if completed_trials >= args.trials:
            break

    study.save()
    best_result_raw = study.get_best_result()
    best_result = _clean_best_result_dict(best_result_raw)

    results_csv = os.path.join(output_dir, "results.csv")
    if os.path.exists(results_csv):
        print(f"Full trial history stored at: {results_csv}")

    summary_path = os.path.join(output_dir, "sherpa_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(best_result, file_obj, indent=2)

    print("\n=== Sherpa optimisation complete ===")
    print(f"Results directory: {output_dir}")
    print(f"Best result: {best_result}")
    print(f"Summary saved to: {summary_path}")


def main() -> None:
    args = _parse_arguments()
    optimise_with_sherpa(args)


if __name__ == "__main__":
    main()
