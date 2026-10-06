"""Direct CSV-driven loading; input errors propagate to stop training."""

from pathlib import Path

from torch.utils.data import Dataset

from brainIAC_pretraining._common import read_ids


class NiftiDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform, transform_prime=None):
        self.ids = read_ids(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.transform_prime = transform if transform_prime is None else transform_prime

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_path = str(self.root_dir / f"{self.ids[index]}.nii.gz")
        # Separate calls sample independent augmentation parameters.
        x0 = self.transform({"image": image_path})
        x1 = self.transform_prime({"image": image_path})
        return x0, x1
