from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from .data import TrajectoryDataset
from .losses import WeightedL1Loss
from .model import MultiLevelPENet
from .sam import SAM
from .utils import device_from_config, ensure_dir, environment_snapshot, parameter_weights, save_json, set_seed


DEFAULT_OPTIMIZER = "adam"
DEFAULT_SAM_BASE_OPTIMIZER = "adam"


def build_model(config: Mapping[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    parameters = config["experiment"]["parameters"]
    architecture = str(model_cfg.get("architecture", "mlpenet")).lower()
    common = dict(
        input_dim=int(model_cfg.get("input_dim", 1)),
        output_dim=len(parameters),
        hidden_dim=int(model_cfg.get("hidden_dim", 25)),
        lstm_layers=int(model_cfg.get("lstm_layers", 4)),
        fc_layers=int(model_cfg.get("fc_layers", 3)),
        fc_hidden=int(model_cfg.get("fc_hidden", 20)),
        activation=str(model_cfg.get("activation", "elu")),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    if architecture == "mlpenet":
        return MultiLevelPENet(
            **common,
            conv_layers=int(model_cfg.get("conv_layers", 2)),
            conv_channels=int(model_cfg.get("conv_channels", 25)),
            conv_kernel_size=int(model_cfg.get("conv_kernel_size", 3)),
            split_parts=int(model_cfg.get("split_parts", 8)),
        )
    raise ValueError(f"Unknown model.architecture: {architecture}")


def _bn_modules(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            yield module


def enable_running_stats(model) -> None:
    for module in _bn_modules(model):
        module.train()
        module.track_running_stats = True


def disable_running_stats(model) -> None:
    for module in _bn_modules(model):
        module.eval()
        module.track_running_stats = False


def _make_optimizer(model: torch.nn.Module, config: Mapping[str, Any]):
    train_cfg = config["training"]
    lr = float(train_cfg.get("learning_rate", 1e-3))
    optimizer_name = str(train_cfg.get("optimizer", DEFAULT_OPTIMIZER)).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(train_cfg.get("momentum", 0.9)))
    if optimizer_name == "sam":
        base_name = str(train_cfg.get("sam_base_optimizer", DEFAULT_SAM_BASE_OPTIMIZER)).lower()
        if base_name == "sgd":
            base_optimizer = torch.optim.SGD
        elif base_name == "adam":
            base_optimizer = torch.optim.Adam
        else:
            raise ValueError(f"Unknown SAM base optimizer: {base_name}")
        kwargs = {"lr": lr}
        if base_name == "sgd":
            kwargs["momentum"] = float(train_cfg.get("momentum", 0.9))
        return SAM(model.parameters(), base_optimizer, rho=float(train_cfg.get("sam_rho", 0.05)), **kwargs)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y, lengths, h in loader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            h = h.to(device)
            pred = model(x, lengths, h)
            loss = criterion(pred, y)
            total += float(loss.item()) * x.shape[0]
            count += x.shape[0]
    return total / max(count, 1)


def train(config: Mapping[str, Any], data_dir: str | Path, run_dir: str | Path) -> Path:
    set_seed(int(config.get("seed", 0)), deterministic=bool(config.get("deterministic", True)))
    run_dir = ensure_dir(run_dir)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "environment.json", environment_snapshot())

    train_set = TrajectoryDataset(Path(data_dir) / "train")
    eval_set = TrajectoryDataset(Path(data_dir) / "eval")
    train_cfg = config["training"]
    device = device_from_config(train_cfg.get("device", "auto"))
    model = build_model(config).to(device)

    weights = parameter_weights(config["experiment"]["parameters"], train_cfg.get("loss_weights"))
    criterion = WeightedL1Loss(weights).to(device)
    optimizer = _make_optimizer(model, config)
    optimizer_name = str(train_cfg.get("optimizer", DEFAULT_OPTIMIZER)).lower()
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=bool(train_cfg.get("drop_last", True)),
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 256))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "eval_loss", "seconds"])
        writer.writeheader()

    best_eval = float("inf")
    best_path = checkpoint_dir / "best.pt"
    epochs = int(train_cfg.get("epochs", 100))
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        train_loss = 0.0
        train_seen = 0
        for x, y, lengths, h in train_loader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            h = h.to(device)
            if optimizer_name == "sam":
                optimizer.zero_grad()
                enable_running_stats(model)
                pred = model(x, lengths, h)
                loss = criterion(pred, y)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                pred = model(x, lengths, h)
                second_loss = criterion(pred, y)
                second_loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                pred = model(x, lengths, h)
                loss = criterion(pred, y)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            train_loss += float(loss.item()) * x.shape[0]
            train_seen += x.shape[0]

        train_loss = train_loss / max(train_seen, 1)
        eval_loss = _evaluate_loss(model, eval_loader, criterion, device)
        seconds = time.time() - start
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "eval_loss", "seconds"])
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.8g}",
                    "eval_loss": f"{eval_loss:.8g}",
                    "seconds": f"{seconds:.3f}",
                }
            )
        print(
            f"epoch {epoch:04d}/{epochs} train={train_loss:.6g} eval={eval_loss:.6g} seconds={seconds:.1f}",
            flush=True,
        )

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": dict(config),
            "parameter_names": list(config["experiment"]["parameters"].keys()),
            "eval_loss": eval_loss,
        }
        torch.save(payload, checkpoint_dir / "last.pt")
        if eval_loss < best_eval:
            best_eval = eval_loss
            torch.save(payload, best_path)

    return best_path
