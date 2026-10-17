"""Check pretraining imports and synthetic data; no patient data or weights loaded."""

import importlib
from importlib.metadata import version
from pathlib import Path
import sys
import tempfile


def main():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Python 3.10 required; running {sys.version}")
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "src/preprocessing"))

    import nibabel as nib
    import numpy as np
    import torch
    from lightly.loss import NTXentLoss
    from simclr.dataset import NiftiDataset
    from simclr.train_multigpu import build_transform

    for name in ("torch", "torchvision", "numpy", "monai", "pytorch-lightning", "lightly", "setuptools"):
        print(f"{name}: {version(name)}")
    for name in ("manifest", "quality", "quality_v2", "convert", "pre_process", "exclude", "exclude_v2", "train"):
        importlib.import_module(f"brainIAC_pretraining.{name}")
    for name in ("simclr.model", "brainiac_embedding._preprocess_bridge",
                 "HD_BET.run", "HD_BET.config", "mri_preprocess_3d_simple"):
        importlib.import_module(name)

    # Exercise real NIfTI loading, two augmentations and contrastive loss.
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        unique_id = "environment_1.2.3_1"
        image = root / f"{unique_id}.nii.gz"
        array = np.random.default_rng(0).normal(size=(16, 16, 16)).astype(np.float32)
        nib.save(nib.Nifti1Image(array, np.eye(4)), image)
        csv_file = root / "final_pre_process.csv"
        csv_file.write_text(f"unique_id\n{unique_id}\n", encoding="utf-8")
        samples = NiftiDataset(csv_file, root, build_transform())[0]
        for sample in samples:
            assert tuple(sample["image"].shape) == (1, 96, 96, 96)
            assert torch.isfinite(sample["image"]).all()
    features = torch.randn(2, 8, requires_grad=True)
    loss = NTXentLoss()(features, torch.randn(2, 8))
    assert torch.isfinite(loss)
    loss.backward()
    print("Pretraining imports, NIfTI transforms and contrastive loss: OK")

    for relative in ("src/preprocessing/atlases/temp_head.nii.gz", "src/preprocessing/hd-bet_params/0.model"):
        if not (repo_root / relative).is_file():
            raise FileNotFoundError(f"Required preprocessing resource missing: {repo_root / relative}")
    print(f"CUDA build: {torch.version.cuda}; CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        value = torch.ones(2, device="cuda")
        assert (value + value).sum().item() == 4
        print(f"GPU tensor operation: OK ({torch.cuda.get_device_name(0)})")
    else:
        print("GPU check skipped: rerun this script on your allocated GPU node before GPU training.")


if __name__ == "__main__":
    main()
