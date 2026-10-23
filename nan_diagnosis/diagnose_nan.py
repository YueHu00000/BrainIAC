"""Standalone, read-only-source BrainIAC NaN investigation: inspect / run / replay."""

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.nn import functional as F


class NonFiniteError(RuntimeError):
    pass


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().as_subclass(torch.Tensor).cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return value


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False,
                                     indent=2, default=str, allow_nan=False), encoding="utf-8")


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def environment():
    versions = {}
    for name in ("torch", "monai", "lightly", "pytorch-lightning", "nibabel", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "versions": versions, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def tensor_stats(value):
    value = value.detach().as_subclass(torch.Tensor)
    finite = torch.isfinite(value)
    result = {"shape": list(value.shape), "dtype": str(value.dtype),
              "finite": bool(finite.all()), "nonfinite": int((~finite).sum())}
    if value.numel() and result["finite"]:
        floating = value.float()
        variance, mean = torch.var_mean(floating, correction=0)
        result.update(max_abs=float(floating.abs().max()), mean=float(mean),
                      std=float(variance.sqrt()),
                      norm=float(torch.linalg.vector_norm(floating, dtype=torch.float64)))
        if value.ndim == 2:
            norms = torch.linalg.vector_norm(floating, dim=1, dtype=torch.float64)
            result.update(vector_norm_min=float(norms.min()), vector_norm_max=float(norms.max()))
    elif not result["finite"]:
        result.update(nan_count=int(torch.isnan(value).sum()), inf_count=int(torch.isinf(value).sum()))
    return result


def tensors_in(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors_in(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors_in(item, f"{prefix}.{index}")


def nonfinite_tensors(items):
    """One host synchronization per device for the common all-finite case."""
    groups = {}
    for name, value in items:
        groups.setdefault(value.device, []).append((name, value))
    issues = []
    for group in groups.values():
        flags = torch.stack([torch.isfinite(value).all() for _, value in group])
        for index in (~flags).nonzero().flatten().cpu().tolist():
            name, value = group[index]
            issues.append({"name": name, **tensor_stats(value)})
    return issues


def nonfinite_scalars(value, prefix=""):
    issues = []
    if isinstance(value, float) and not math.isfinite(value):
        issues.append({"name": prefix, "value": str(value)})
    elif isinstance(value, dict):
        for key, item in value.items():
            issues.extend(nonfinite_scalars(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            issues.extend(nonfinite_scalars(item, f"{prefix}.{index}"))
    return issues


def gradient_stats(model):
    grads = [(name, parameter.grad) for name, parameter in model.named_parameters()
             if parameter.grad is not None]
    issues = nonfinite_tensors(grads)
    norm = max_abs = None
    if grads and not issues:
        norms = torch.stack([torch.linalg.vector_norm(g.detach(), dtype=torch.float64)
                             for _, g in grads])
        norm = float(torch.linalg.vector_norm(norms))
        max_abs = float(torch.stack([g.detach().abs().max().float() for _, g in grads]).max())
    return {"finite": not issues, "norm": norm, "max_abs": max_abs, "issues": issues}


def forward_views(model, batch, observer=None):
    outputs = []
    for view, data in enumerate(batch):
        def check(stage, value):
            if observer is not None:
                observer(f"view{view}.{stage}", value)
            return value

        image = check("input", data["image"])
        tokens, _ = model.backbone(image)
        tokens = check("backbone", tokens)
        pooled = check("pooled", tokens.mean(dim=1))
        projected = check("projection", model.projection_head(pooled))
        if observer is not None and hasattr(observer, "check_buffers"):
            observer.check_buffers(f"view{view}.buffers")
        outputs.append(check("normalize", F.normalize(projected, dim=1)))
    loss = model.criterion(*outputs)
    if observer is not None:
        observer("loss", loss)
    return loss


def make_model(hparams, probe=None):
    """Same parameter names and math as the source; no source-project imports."""
    import pytorch_lightning as pl
    from lightly.loss import NTXentLoss
    from lightly.models.modules.heads import SimCLRProjectionHead
    from monai.networks.nets import ViT

    class DiagnosticModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.save_hyperparameters(hparams)
            self.backbone = ViT(
                in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
                hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
                classification=False, save_attn=True)
            self.projection_head = SimCLRProjectionHead(768, 768, 2048)
            self.criterion = NTXentLoss()

        def forward(self, x):
            tokens, _ = self.backbone(x)
            return F.normalize(self.projection_head(tokens.mean(dim=1)), dim=1)

        def training_step(self, batch, batch_idx):
            loss = forward_views(self, batch, probe)
            self.log("train_loss_ssl", loss, on_step=True, on_epoch=True,
                     batch_size=batch[0]["image"].shape[0])
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                          weight_decay=5e-4, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=1, eta_min=1e-6)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    return DiagnosticModel()


class CheckData:
    """Deterministic Compose member: does not draw random numbers or reseed transforms."""
    def __init__(self, stage, output_dir):
        self.stage = stage
        self.output_dir = Path(output_dir)

    def __call__(self, data):
        stats = tensor_stats(torch.as_tensor(data["image"]))
        data["trace"].append({"stage": self.stage, **stats})
        if not stats["finite"]:
            worker = torch.utils.data.get_worker_info()
            folder = self.output_dir / f"data_{worker.id if worker else 'main'}_{time.time_ns()}"
            folder.mkdir(parents=True, exist_ok=False)
            write_json(folder / "summary.json", {"kind": "data_nonfinite",
                       "unique_id": data["unique_id"], "path": data["path"],
                       "view": data["view"], "stage": self.stage, "trace": data["trace"],
                       "replay_scope": "observed_tensor_only; no complete model batch exists"})
            torch.save(cpu_copy(data["image"]), folder / "tensor.pt")
            raise NonFiniteError(f"NaN diagnosis stopped: {data['unique_id']} {self.stage}; {folder}")
        return data


def build_transform(output_dir=None):
    from monai.transforms import (
        Compose, LoadImaged, NormalizeIntensityd, RandAdjustContrastd, RandAffined,
        RandBiasFieldd, RandCoarseDropoutd, RandFlipd, RandGaussianNoised,
        RandGaussianSmoothd, RandScaleCropd, Resized, ToTensord, Transform)

    class RandomResizedCrop3D(Transform):
        # Exact source behavior, including construction of the inner random crop.
        def __call__(self, image):
            if torch.rand(1) < 1.0:
                scale = torch.empty(1).uniform_(0.2, 1.0).item()
                for transform in (RandScaleCropd(["image"], scale, random_size=False),
                                  Resized(["image"], (96, 96, 96), mode="trilinear")):
                    image = transform(image)
            return image

    keys = ["image"]
    transforms = [
        LoadImaged(keys=keys, ensure_channel_first=True),
        Resized(keys=keys, spatial_size=(96, 96, 96)),
        RandBiasFieldd(keys=keys, prob=0.3),
        NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        RandomResizedCrop3D(),
        RandFlipd(keys=keys, spatial_axis=[0], prob=0.5),
        RandAffined(keys=keys, rotate_range=(np.pi / 12, np.pi / 12, 0),
                    translate_range=((-10, 10), (-10, 10), (-5, 5)),
                    scale_range=((0.85, 1.15), (0.85, 1.15), (0.9, 1.1)),
                    padding_mode="border", prob=0.7),
        RandGaussianSmoothd(keys=keys, prob=0.5, sigma_x=(0.5, 1.5),
                            sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        RandCoarseDropoutd(keys=keys, prob=0.5, holes=2, spatial_size=(2, 2, 2),
                          max_spatial_size=(4, 4, 4), fill_value=0.0),
        RandGaussianNoised(keys=keys, prob=0.5, std=0.05),
        RandAdjustContrastd(keys=keys, prob=0.5, gamma=(0.7, 1.3)),
        ToTensord(keys=keys)]
    if output_dir is not None:
        transforms = [item for index, transform in enumerate(transforms)
                      for item in (transform, CheckData(f"{index:02d}.{type(transform).__name__}", output_dir))]
    return Compose(transforms)


class DiagnosticDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, input_dir, output_dir):
        with Path(csv_file).open(encoding="utf-8-sig", newline="") as stream:
            self.ids = [row["unique_id"] for row in csv.DictReader(stream)]
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.transform = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Construct in the worker on spawn platforms, but run() initializes it in
        # the parent on Linux to preserve the original fork/worker initialization.
        if self.transform is None:
            self.transform = build_transform(self.output_dir)
        unique_id = self.ids[index]
        path = str(self.input_dir / f"{unique_id}.nii.gz")
        views = []
        for view in range(2):
            data = self.transform({"image": path, "unique_id": unique_id, "path": path,
                                   "view": view, "trace": []})
            views.append({"image": data["image"], "unique_id": unique_id, "path": path,
                          "trace": json.dumps(data["trace"])})
        return tuple(views)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["transform"] = None
        return state


class Probe:
    def __init__(self, output_dir, capture_update_state=False):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capture_update_state = capture_update_state
        self.captured = set()
        self.updates = 0

    def event(self, value):
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False) + "\n")

    def begin(self, model, batch, optimizer, scheduler, scaler, context, precision):
        self.model, self.batch = model, batch
        self.optimizer, self.scheduler, self.scaler = optimizer, scheduler, scaler
        self.context, self.precision = context, precision
        self.stages = {}
        for view_index, view in enumerate(batch):
            grouped = {}
            for trace in view.get("trace", []):
                for entry in json.loads(trace):
                    grouped.setdefault(entry["stage"], []).append(entry)
            for stage, entries in grouped.items():
                self.stages[f"data.view{view_index}.{stage}"] = {
                    "finite": all(entry["finite"] for entry in entries),
                    "sample_count": len(entries),
                    "max_abs": max(entry.get("max_abs", 0.) for entry in entries),
                    "sample_std_min": min(entry.get("std", 0.) for entry in entries),
                    "sample_std_max": max(entry.get("std", 0.) for entry in entries)}
        self.grad = {"finite": True, "norm": None, "max_abs": None, "issues": []}
        self.before_updates = self.updates
        self.before_scale = float(scaler.get_scale()) if scaler is not None else None
        self.before_rng = rng_state()
        self.before_buffers = cpu_copy(dict(model.named_buffers()))
        self.before_update = self.snapshot() if self.capture_update_state else None

    def snapshot(self, post_update=False):
        state = cpu_copy(self.model.state_dict())
        buffers = (cpu_copy(dict(self.model.named_buffers())) if post_update
                   else cpu_copy(self.before_buffers))
        if not post_update:
            # Lightly has nonpersistent memory-bank buffers; never insert them
            # into state_dict, otherwise strict loading rejects the scene.
            state.update({name: value for name, value in buffers.items() if name in state})
        return {"format_version": 1, "model_state": state, "buffers": buffers,
                "hparams": dict(getattr(self.model, "hparams", {})),
                "optimizer_state": cpu_copy(self.optimizer.state_dict()),
                "scheduler_state": cpu_copy(self.scheduler.state_dict()) if self.scheduler else None,
                "scaler_state": cpu_copy(self.scaler.state_dict()) if self.scaler is not None else None,
                "rng": cpu_copy(self.before_rng), "training": self.model.training,
                "precision": self.precision, "environment": environment(),
                "model_device": str(next(self.model.parameters()).device),
                "state_phase": "post_update" if post_update else "pre_forward"}

    def capture(self, kind, stage, post_update=False):
        if kind in self.captured:
            return
        self.captured.add(kind)
        folder = self.root / f"first_{kind}"
        folder.mkdir(exist_ok=False)
        state = self.snapshot(post_update)
        scope = "forward_backward"
        if post_update:
            if self.before_update is not None:
                torch.save(state, folder / "post_update_state.pt")
                state = self.before_update
                scope = "forward_backward_update"
            else:
                scope = "post_update_inspection_only"
        elif self.capture_update_state:
            scope = "forward_backward_update"
        summary = {"kind": kind, "stage": stage, **self.context,
                   "unique_ids": list(self.batch[0].get("unique_id", [])),
                   "stages": self.stages, "gradients": self.grad,
                   "replay_scope": scope, "state_phase": state["state_phase"]}
        torch.save(cpu_copy(self.batch), folder / "batch.pt")
        torch.save(state, folder / "state.pt")
        write_json(folder / "summary.json", summary)
        self.event({"event": "capture", "folder": str(folder), **summary})

    def __call__(self, stage, value):
        stats = tensor_stats(value)
        self.stages[stage] = stats
        if not stats["finite"]:
            self.capture("forward_nonfinite", stage)
            raise NonFiniteError(f"NaN diagnosis stopped: {stage}; {self.root}")

    def check_buffers(self, stage):
        issues = nonfinite_tensors(self.model.named_buffers())
        if issues:
            self.stages[stage] = {"finite": False, "issues": issues}
            self.capture("buffer_nonfinite", stage)
            raise NonFiniteError(f"NaN diagnosis stopped: {stage}")

    def check_gradients(self):
        self.grad = gradient_stats(self.model)
        if not self.grad["finite"]:
            self.capture("gradient_nonfinite", "unscaled_preclip_gradients")
            if self.scaler is None:
                raise NonFiniteError("NaN diagnosis stopped: nonfinite gradients without AMP scaler")

    def finish(self):
        issues = nonfinite_tensors(
            list(self.model.named_parameters()) + list(self.model.named_buffers())
            + list(tensors_in(self.optimizer.state_dict(), "optimizer")))
        row = {**self.context, "loss": self.stages.get("loss", {}).get("mean"),
               "grad_finite": self.grad["finite"], "grad_norm": self.grad["norm"],
               "grad_max_abs": self.grad["max_abs"], "scale_before": self.before_scale,
               "scale_after": float(self.scaler.get_scale()) if self.scaler is not None else None,
               "optimizer_updated": self.updates > self.before_updates,
               "state_finite": not issues}
        path = self.root / "steps.csv"
        new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        # Per-stage scalar statistics are separate from the compact CSV.
        self.event({"event": "batch", **row, "unique_ids": list(self.batch[0].get("unique_id", [])),
                    "stages": self.stages, "gradient_issues": self.grad["issues"]})
        if issues:
            self.stages["post_update_state"] = {"finite": False, "issues": issues}
            self.capture("update_nonfinite", "post_update_state", post_update=True)
            raise NonFiniteError("NaN diagnosis stopped: post-update state")
        self.before_update = None
        self.batch = None


def make_callback(probe):
    import pytorch_lightning as pl

    class DiagnosticCallback(pl.Callback):
        def on_fit_start(self, trainer, model):
            self.handle = trainer.optimizers[0].register_step_post_hook(self.updated)

        def updated(self, optimizer, args, kwargs):
            probe.updates += 1

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            scheduler = trainer.lr_scheduler_configs[0].scheduler
            probe.begin(model, batch, trainer.optimizers[0], scheduler,
                        getattr(trainer.precision_plugin, "scaler", None),
                        {"epoch": trainer.current_epoch, "batch_idx": batch_idx,
                         "global_step": trainer.global_step,
                         "lr": trainer.optimizers[0].param_groups[0]["lr"]}, trainer.precision)

        def on_before_optimizer_step(self, trainer, model, optimizer):
            probe.check_gradients()

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            probe.finish()

        def on_fit_end(self, trainer, model):
            self.handle.remove()

        def on_exception(self, trainer, model, exception):
            if hasattr(self, "handle"):
                self.handle.remove()

    return DiagnosticCallback()


def inspect_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    model_issues = nonfinite_tensors(tensors_in(state))
    optimizer_issues = nonfinite_tensors(tensors_in(checkpoint.get("optimizer_states", [])))
    optimizer_issues += nonfinite_scalars(checkpoint.get("optimizer_states", []), "optimizer")
    scheduler_issues = nonfinite_scalars(checkpoint.get("lr_schedulers", []), "scheduler")
    scaler_issues = nonfinite_scalars(checkpoint.get("native_amp_scaling_state", {}), "scaler")
    negative_variance = [name for name, value in state.items()
                         if name.endswith("running_var") and bool((value < 0).any())]
    report = {"checkpoint": str(path), "epoch": checkpoint.get("epoch"),
              "global_step": checkpoint.get("global_step"),
              "model_nonfinite": model_issues, "optimizer_nonfinite": optimizer_issues,
              "scheduler_nonfinite": scheduler_issues, "amp_scaler_nonfinite": scaler_issues,
              "negative_running_variance": negative_variance,
              "static_finite": not any((model_issues, optimizer_issues, scheduler_issues,
                                         scaler_issues, negative_variance)),
              "optimizer_present": bool(checkpoint.get("optimizer_states")),
              "lr_schedulers": checkpoint.get("lr_schedulers"),
              "amp_scaler": checkpoint.get("native_amp_scaling_state"),
              "unchecked": ["fixed_sample_forward", "historical_training_trajectory"]}
    return checkpoint, report


def inspect_command(args):
    output = fresh_output(args.output_dir)
    checkpoint, report = inspect_checkpoint(args.checkpoint)
    if args.sample:
        from monai.transforms import Compose, LoadImaged, NormalizeIntensityd, Resized, ToTensord
        transform = Compose([LoadImaged(["image"], ensure_channel_first=True),
                             Resized(["image"], (96, 96, 96)),
                             NormalizeIntensityd(["image"], nonzero=True, channel_wise=True),
                             ToTensord(["image"])])
        image = torch.stack([torch.as_tensor(transform({"image": str(p)})["image"])
                             for p in args.sample]).to(args.device)
        initial_rng = rng_state()
        results = []
        # Fresh model/state for each mode; original checkpoint is never rewritten.
        for training in (True, False):
            model = make_model(checkpoint.get("hyper_parameters", {"lr": 5e-4}))
            model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
            model.to(args.device).train(training)
            restore_rng(initial_rng)
            stages = {}

            def observe(stage, value):
                stages[stage] = tensor_stats(value)
                if not stages[stage]["finite"]:
                    raise NonFiniteError(stage)

            def check_buffers(stage):
                issues = nonfinite_tensors(model.named_buffers())
                if issues:
                    stages[stage] = {"finite": False, "issues": issues}
                    raise NonFiniteError(stage)

            observe.check_buffers = check_buffers

            try:
                with torch.no_grad():
                    forward_views(model, ({"image": image}, {"image": image}), observe)
                results.append({"training": training, "finite": True, "stages": stages})
            except NonFiniteError as error:
                results.append({"training": training, "finite": False,
                                "first_nonfinite": str(error), "stages": stages})
            del model
        report["fixed_sample_forward"] = results
        report["unchecked"].remove("fixed_sample_forward")
        report["fixed_sample_precision"] = "32-true; resized and normalized images without random augmentation"
    write_json(output / "checkpoint_inspection.json", report)
    write_json(output / "environment.json", environment())
    print(json.dumps(json_safe(report), ensure_ascii=False, default=str))
    return 0 if report["static_finite"] and all(r["finite"] for r in report.get("fixed_sample_forward", [])) else 2


def fresh_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_command(args):
    import pytorch_lightning as pl
    import yaml
    from monai.utils import set_determinism
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    config = {"data": {"batch_size": 160, "num_workers": 24, "size": [96, 96, 96]},
              "optim": {"lr": 5e-4}, "model": {"max_epochs": 200}}
    if args.config:
        with args.config.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    for name in ("batch_size", "num_workers"):
        if getattr(args, name) is not None:
            config["data"][name] = getattr(args, name)
    if args.max_epochs is not None:
        config["model"]["max_epochs"] = args.max_epochs
    if tuple(config["data"]["size"]) != (96, 96, 96):
        raise ValueError("This one-off diagnosis reproduces the original 96x96x96 input only")
    if args.resume:
        checkpoint, report = inspect_checkpoint(args.resume)
        if not report["static_finite"]:
            raise ValueError("Starting checkpoint is already nonfinite; run inspect first")
        if not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers"):
            raise ValueError("run --resume requires a full training checkpoint, not encoder weights")
        del checkpoint
    else:
        report = {"initialization": "scratch", "checkpoint_checked": False}
    output = fresh_output(args.output_dir)
    set_determinism(seed=0)
    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")
    dataset = DiagnosticDataset(args.final_csv, args.input_dir, output / "diagnostics")
    dataset.transform = build_transform(output / "diagnostics")
    batch_size = config["data"]["batch_size"]
    if batch_size < 2 or len(dataset) < batch_size:
        raise ValueError("Need batch_size >= 2 and at least one full batch")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                drop_last=True, num_workers=config["data"]["num_workers"],
                pin_memory=args.accelerator != "cpu")
    probe = Probe(output / "diagnostics", args.capture_update_state)
    model = make_model({"lr": config["optim"]["lr"],
                        "max_epochs": config["model"]["max_epochs"]}, probe)
    gpu = args.accelerator == "gpu" or (args.accelerator == "auto" and torch.cuda.is_available())
    precision = args.precision or ("16-mixed" if gpu else "32-true")
    if not gpu and precision != "32-true":
        raise ValueError("FP16/BF16 diagnosis requires CUDA; use 32-true on CPU")
    config["diagnosis"] = {"resume": str(args.resume) if args.resume else None,
                          "precision": precision, "capture_update_state": args.capture_update_state,
                          "final_csv": str(args.final_csv), "input_dir": str(args.input_dir),
                          "sample_count": len(dataset), "max_steps_absolute": args.max_steps,
                          "gradient_clip_val": 2.0, "weight_decay": 5e-4,
                          "temperature": model.criterion.temperature,
                          "representation": "all_216_patch_tokens_mean_v1"}
    with (output / "run_config.yml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    write_json(output / "checkpoint_inspection.json", report)
    write_json(output / "environment.json", environment())
    checkpoint_callback = ModelCheckpoint(dirpath=output / "checkpoints", save_last=True,
                                          save_top_k=0, save_weights_only=False)
    trainer = pl.Trainer(max_epochs=config["model"]["max_epochs"], max_steps=args.max_steps,
                         accelerator=args.accelerator,
                         devices=[args.device] if gpu and args.device is not None else 1,
                         precision=precision, gradient_clip_val=2.0,
                         logger=CSVLogger(str(output), name="logs"),
                         callbacks=[make_callback(probe), checkpoint_callback])
    trainer.fit(model, train_dataloaders=loader,
                ckpt_path=str(args.resume) if args.resume else None)
    return 0


def replay_trial(state, batch, precision, device, do_update=False, anomaly_detection=False,
                 model_factory=make_model, forward_only=False):
    if state["state_phase"] != "pre_forward":
        raise ValueError("Post-update-only state cannot reproduce the original failing forward/update")
    model = model_factory(state["hparams"])
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).train(state["training"])
    for name, value in state["buffers"].items():
        parent, _, key = name.rpartition(".")
        model.get_submodule(parent)._buffers[key] = value.to(device).clone()
    batch = tuple({"image": view["image"].to(device)} for view in batch)
    optimizer = None
    if not forward_only:
        optim_config = model.configure_optimizers()
        optimizer = optim_config["optimizer"]
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler = optim_config["lr_scheduler"]["scheduler"]
        if state["scheduler_state"] is not None:
            scheduler.load_state_dict(state["scheduler_state"])
    scaler = torch.amp.GradScaler("cuda") if precision == "16-mixed" and not forward_only else None
    if scaler is not None and state["scaler_state"] is not None:
        scaler.load_state_dict(state["scaler_state"])
    # Setting allow_tf32 afterwards can overwrite 'medium' with 'high' (and
    # mixes legacy/new precision APIs on recent torch). Restore one API only.
    torch.set_float32_matmul_precision(state["environment"]["matmul_precision"])
    torch.backends.cudnn.allow_tf32 = state["environment"]["cudnn_allow_tf32"]
    restore_rng(state["rng"])
    if device.startswith("cuda") and state["rng"]["cuda"]:
        source_index = torch.device(state["model_device"]).index or 0
        torch.cuda.set_rng_state(state["rng"]["cuda"][source_index].cpu(), device)
    stages = {}
    result = {"precision": precision, "stages": stages, "optimizer_updated": False}

    def observe(stage, value):
        stages[stage] = tensor_stats(value)
        if not stages[stage]["finite"]:
            result["first_nonfinite"] = stage
            raise NonFiniteError(stage)

    def check_buffers(stage):
        issues = nonfinite_tensors(model.named_buffers())
        if issues:
            result.update(first_nonfinite=stage, buffer_issues=issues)
            raise NonFiniteError(stage)

    observe.check_buffers = check_buffers
    def updated(*unused):
        result["optimizer_updated"] = True
    handle = optimizer.register_step_post_hook(updated) if optimizer is not None else None
    autocast = (torch.autocast("cuda", dtype=torch.float16 if precision == "16-mixed" else torch.bfloat16)
                if precision != "32-true" else nullcontext())
    phase = "forward"
    try:
        with torch.autograd.detect_anomaly() if anomaly_detection else nullcontext():
            with torch.no_grad() if forward_only else nullcontext(), autocast:
                loss = forward_views(model, batch, observe)
            if forward_only:
                result.update(finite=True, backward_ran=False)
                return result
            result["backward_ran"] = True
            phase = "backward"
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            result["gradients"] = gradient_stats(model)
            if not result["gradients"]["finite"]:
                result["first_nonfinite"] = "unscaled_preclip_gradients"
            if do_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                if scaler is None:
                    if result["gradients"]["finite"]:
                        optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                issues = nonfinite_tensors(list(model.named_parameters()) + list(model.named_buffers())
                                           + list(tensors_in(optimizer.state_dict(), "optimizer")))
                result["post_update_issues"] = issues
                if issues and "first_nonfinite" not in result:
                    result["first_nonfinite"] = "post_update_state"
    except NonFiniteError:
        pass
    except RuntimeError as error:
        # Anomaly detection can throw before the gradient summary is available.
        if not (anomaly_detection and phase == "backward" and "returned nan values" in str(error)):
            raise
        result["backward_exception"] = str(error)
    finally:
        if handle is not None:
            handle.remove()
    result["finite"] = "first_nonfinite" not in result and "backward_exception" not in result
    return result


def replay_command(args):
    output = fresh_output(args.output_dir)
    folder = args.scene
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["kind"] == "data_nonfinite":
        stats = tensor_stats(torch.load(folder / "tensor.pt", map_location="cpu", weights_only=False))
        report = {"scene": str(folder), "scope": "observed_data_tensor_only",
                  "stage": summary["stage"], "unique_id": summary["unique_id"], "stats": stats,
                  "note": "No re-execution of augmentation or model; see saved transform trace."}
        write_json(output / "replay_report.json", report)
        print(json.dumps(report))
        return 0
    state = torch.load(folder / "state.pt", map_location="cpu", weights_only=False)
    batch = torch.load(folder / "batch.pt", map_location="cpu", weights_only=False)
    if args.update and summary["replay_scope"] != "forward_backward_update":
        raise ValueError("Exact update replay needs a scene captured with --capture-update-state")
    results = []
    for precision in args.precisions:
        if precision != "32-true" and (not args.device.startswith("cuda") or not torch.cuda.is_available()):
            results.append({"precision": precision, "status": "not_run_requires_cuda"})
            continue
        if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
            results.append({"precision": precision, "status": "not_run_bf16_unsupported"})
            continue
        results.append(replay_trial(state, batch, precision, args.device,
                                    args.update, args.detect_anomaly,
                                    forward_only=args.forward_only))
    report = {"scene": str(folder), "source_summary": summary, "trials": results,
              "source_environment": state["environment"],
              "environment": environment(),
              "note": "Same tensors, initial state, mode and RNG. FP32 keeps saved TF32 settings."}
    write_json(output / "replay_report.json", report)
    print(json.dumps(json_safe(results), ensure_ascii=False))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Check checkpoint without changing it")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inspect.add_argument("--sample", type=Path, action="append", help="Repeat for >=2 fixed NIfTI inputs")
    inspect.add_argument("--device", default="cpu")
    inspect.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run", help="Capture NaN under original training conditions")
    run.add_argument("--final-csv", type=Path, required=True)
    run.add_argument("--input-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--resume", type=Path)
    run.add_argument("--config", type=Path)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--num-workers", type=int)
    run.add_argument("--max-epochs", type=int)
    run.add_argument("--max-steps", type=int, default=-1, help="Absolute Lightning global-step limit")
    run.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    run.add_argument("--device", type=int)
    run.add_argument("--precision", choices=("16-mixed", "32-true", "bf16-mixed"))
    run.add_argument("--capture-update-state", action="store_true",
                     help="Expensive: CPU copy of full pre-forward model/optimizer every batch")
    replay = commands.add_parser("replay", help="Replay saved tensors/state, not random augmentations")
    replay.add_argument("--scene", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--device", default="cuda:0")
    replay.add_argument("--precisions", nargs="+", default=["16-mixed", "32-true"],
                        choices=("16-mixed", "32-true", "bf16-mixed"))
    replay.add_argument("--update", action="store_true")
    replay.add_argument("--detect-anomaly", action="store_true")
    replay.add_argument("--forward-only", action="store_true", help="Skip backward to reduce memory")
    args = parser.parse_args(argv)
    if args.command == "inspect" and args.sample and len(args.sample) < 2:
        parser.error("--sample needs >=2 inputs for training-mode projection BatchNorm")
    if args.command == "replay" and args.forward_only and args.update:
        parser.error("--forward-only and --update cannot be used together")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        return {"inspect": inspect_command, "run": run_command, "replay": replay_command}[args.command](args)
    except RuntimeError as error:
        cause = error
        while cause is not None:
            if isinstance(cause, NonFiniteError) or "NaN diagnosis stopped:" in str(cause):
                print(str(cause))
                return 2
            cause = cause.__cause__ or cause.__context__
        raise


if __name__ == "__main__":
    raise SystemExit(main())
