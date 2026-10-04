"""Train SimCLR directly from final_pre_process.csv and its NIfTI directory."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-csv", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "simclr/simclr/config.yml")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, help="Resume full Lightning checkpoint")
    initialization.add_argument("--init-checkpoint", type=Path, help="Initialize encoder weights only")
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--device", type=int, help="Single GPU index")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args(argv)


def run(args):
    import pytorch_lightning as pl
    import torch
    import yaml
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader
    from monai.utils import set_determinism
    from simclr.simclr.dataset import NiftiDataset
    from simclr.simclr.model import SimCLRModel
    from simclr.simclr.train_multigpu import build_transform

    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for name in ("batch_size", "num_workers"):
        value = getattr(args, name)
        if value is not None:
            config["data"][name] = value
    if args.max_epochs is not None:
        config["model"]["max_epochs"] = args.max_epochs
    set_determinism(seed=0)
    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")
    dataset = NiftiDataset(args.final_csv, args.input_dir,
                          build_transform(config["data"]["size"]))
    batch_size = config["data"]["batch_size"]
    if batch_size < 2 or len(dataset) < batch_size:
        raise ValueError("SimCLR needs batch_size >= 2 and at least one full batch; reduce --batch-size")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=config["data"]["num_workers"],
                        pin_memory=args.accelerator != "cpu")
    model = SimCLRModel({"lr": config["optim"]["lr"],
                         "max_epochs": config["model"]["max_epochs"]})
    if args.init_checkpoint is not None:
        model.initialize_encoder(args.init_checkpoint)
    config["optim"].update(weight_decay=5e-4, temperature=model.criterion.temperature)
    config["representation"] = "all_216_patch_tokens_mean_v1"
    config["inputs"] = {"final_csv": str(args.final_csv), "input_dir": str(args.input_dir),
                        "sample_count": len(dataset)}
    config["initialization"] = {"resume": str(args.resume) if args.resume else None,
                                "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = "run_config.yml" if args.resume is None else "resume_config.yml"
    with (args.output_dir / report).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints", monitor="train_loss_ssl", mode="min",
        save_top_k=10, save_last=True, save_weights_only=False,
        filename="epoch={epoch:03d}-loss={train_loss_ssl:.4f}", auto_insert_metric_name=False,
    )
    gpu = args.accelerator == "gpu" or (args.accelerator == "auto" and torch.cuda.is_available())
    trainer = pl.Trainer(
        max_epochs=config["model"]["max_epochs"], accelerator=args.accelerator,
        devices=[args.device] if gpu and args.device is not None else 1,
        precision="16-mixed" if gpu else "32-true", gradient_clip_val=2.0,
        logger=CSVLogger(str(args.output_dir), name="logs"), callbacks=[checkpoint],
    )
    trainer.fit(model, train_dataloaders=loader,
                ckpt_path=str(args.resume) if args.resume is not None else None)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
