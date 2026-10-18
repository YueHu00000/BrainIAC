"""Export independently trained BrainIAC embeddings, one NPZ per study."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import ViT
from monai.transforms import (
    Compose, LoadImaged, NormalizeIntensityd, Resized, ToTensord,
)


BLOCK_NUMBERS = (3, 6, 9, 12)


def read_csv(path, columns):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def check_id(value):
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"Not a plain file ID: {value!r}")


def group_series(final_csv, selected_csv):
    ids = [row["unique_id"] for row in read_csv(final_csv, ["unique_id"])]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("final CSV must contain nonempty, unique volume IDs")
    for unique_id in ids:
        check_id(unique_id)
    wanted = set(ids)
    lookup = {}
    for row in read_csv(selected_csv, ["unique_id", "study_id", "series_description"]):
        unique_id = row["unique_id"]
        if unique_id not in wanted:
            continue
        if unique_id in lookup:
            raise ValueError(f"Duplicate metadata for {unique_id}")
        check_id(row["study_id"])
        lookup[unique_id] = row
    missing = wanted - lookup.keys()
    if missing:
        raise ValueError(f"Missing series metadata: {sorted(missing)}")
    groups = {}
    for unique_id in ids:
        row = lookup[unique_id]
        groups.setdefault(row["study_id"], []).append(row)
    return groups


def load_encoder(checkpoint_path, device):
    encoder = ViT(
        in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
        hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
        classification=False, save_attn=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    backbone = {key[len("backbone."):]: value for key, value in state.items()
                if key.startswith("backbone.")}
    encoder.load_state_dict(backbone, strict=True)
    return encoder.to(device).requires_grad_(False).eval()


def build_transform():
    return Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True),
        Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"], track_meta=False),
    ])


def extract_volume(encoder, image_path, transform, device):
    image = transform({"image": str(image_path)})["image"]
    if tuple(image.shape) != (1, 96, 96, 96) or not torch.isfinite(image).all():
        raise ValueError(f"{image_path}: expected finite [1,96,96,96] image")
    encoder.eval()
    with torch.inference_mode():
        tokens, states = encoder(image.unsqueeze(0).to(device=device, dtype=torch.float32))
        selected = [states[number - 1] for number in BLOCK_NUMBERS]
        if any(tuple(x.shape) != (1, 216, 768) for x in [tokens, *selected]):
            raise ValueError(f"{image_path}: expected [1,216,768] encoder tokens")
        blocks = torch.stack([state.mean(dim=1) for state in selected], dim=1)[0]
        final = tokens.mean(dim=1)[0]
    blocks = blocks.cpu().to(torch.float32).numpy()
    final = final.cpu().to(torch.float32).numpy()
    if not np.isfinite(blocks).all() or not np.isfinite(final).all():
        raise ValueError(f"{image_path}: nonfinite embeddings")
    return blocks, final


def pack_record(description, blocks, final):
    record = np.empty((), dtype=[
        ("series_description", f"U{max(1, len(description))}"),
        ("block_embed", np.float32, (4, 768)),
        ("final_embed", np.float32, (768,)),
    ])
    record["series_description"] = description
    record["block_embed"] = blocks
    record["final_embed"] = final
    return record


def run(args):
    groups = group_series(args.final_csv, args.selected_csv)
    config = {
        "checkpoint": str(args.checkpoint.resolve()),
        "final_csv": str(args.final_csv.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "selected_csv": str(args.selected_csv.resolve()),
        "representation": "all_216_patch_tokens_mean_v1",
        "block_numbers": list(BLOCK_NUMBERS),
        "image_size": [96, 96, 96],
        "resize_mode": "trilinear",
        "normalization": "per_volume_nonzero_zscore",
        "dtype": "float32",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "embedding_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Output configuration differs; use a new output directory")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    pending = []
    for study_id, series in groups.items():
        output = args.output_dir / f"{study_id}.npz"
        if output.exists():
            print(f"[skip] {study_id}", flush=True)
        else:
            pending.append((study_id, series, output))
    if not pending:
        return

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    encoder = load_encoder(args.checkpoint, device)
    transform = build_transform()
    for study_id, series, output in pending:
        print(f"[compute] {study_id}: {len(series)} series", flush=True)
        records = {}
        for row in series:
            unique_id = row["unique_id"]
            blocks, final = extract_volume(
                encoder, args.input_dir / f"{unique_id}.nii.gz", transform, device,
            )
            records[unique_id] = pack_record(row["series_description"], blocks, final)
        temporary = output.with_name(output.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **records)
        temporary.replace(output)
        print(f"[saved] {output}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-csv", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--selected-csv", type=Path, required=True,
                        help="Matching selected_series.csv with study_id and series_description")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Pretraining checkpoint containing backbone.* weights")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", help="For example cuda:0 or cpu; defaults to CUDA if available")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
