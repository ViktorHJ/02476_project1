import typer
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import random

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl

app = typer.Typer()


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        image_size: Tuple[int, int] = (32, 32),
        random_seed: int = 42,
        cuda: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.image_size = image_size
        self.random_seed = random_seed
        self.cuda = cuda

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

        # One generator used everywhere
        self.generator = torch.Generator().manual_seed(self.random_seed)

    def setup(self, stage: Optional[str] = None):
        """Create datasets. Called automatically by Lightning."""

        # Full training dataset
        full_train_dataset = datasets.ImageFolder(
            root=self.data_dir / "train",
            transform=self.transform,
        )

        # Classes and number of classes
        self.classes = full_train_dataset.classes
        self.num_classes = len(self.classes)

        # Train / validation split
        val_size = int(len(full_train_dataset) * self.val_split)
        train_size = len(full_train_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=self.generator,
        )

        # Test dataset
        self.test_dataset = datasets.ImageFolder(
            root=self.data_dir / "test",
            transform=self.transform,
        )

    def _seed_worker(self, worker_id):
        worker_seed = self.random_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=self.generator,
            worker_init_fn=self._seed_worker,
            persistent_workers=self.num_workers > 0,
            pin_memory=True if self.cuda else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            generator=self.generator,
            worker_init_fn=self._seed_worker,
            persistent_workers=self.num_workers > 0,
            pin_memory=True if self.cuda else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            generator=self.generator,
            worker_init_fn=self._seed_worker,
            persistent_workers=self.num_workers > 0,
            pin_memory=True if self.cuda else False,
        )


@app.command()
def download_CIFAKE_dataset(data_dir: str = "data"):
    """Download the CIFake dataset from Kaggle and extract it to the specified directory.

    Args:
        data_dir (str): Directory to download and extract the dataset to.
    """
    data_dir = Path(data_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    else:
        resp = input(
            f"Data directory {data_dir} already exists. Continue and potentially overwrite existing data? (y/n): "
        )
        if resp.lower() != "y" and resp.lower() != "yes":
            print("Download cancelled.")
            return
        print("Clearing existing data...")
        os.system(f"rm -rf {data_dir}/*")

    print("Downloading CIFake dataset...")
    os.system("kaggle datasets download birdy654/cifake-real-and-ai-generated-synthetic-images")
    print("Download complete. Extracting files...")
    os.system(f"unzip cifake-real-and-ai-generated-synthetic-images.zip -d {data_dir}")
    os.system("rm cifake-real-and-ai-generated-synthetic-images.zip")
    print("Extraction complete.")


@app.command()
def create_data_module(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    image_size: Tuple[int, int] = (32, 32),
    random_seed: int = 42,
    cuda: bool = False,
) -> ImageDataModule:
    """Helper function to test an ImageDataModule with specified parameters.

    Args:
        data_dir (str): Directory where the dataset is located.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of worker processes for data loading.
        val_split (float): Fraction of training data to use for validation.
        image_size (Tuple[int, int]): Size to which images will be resized.
        random_seed (int): Random seed for reproducibility.
        cuda (bool): Whether to use CUDA (GPU) or not.

    Returns:
        ImageDataModule: Configured data module."""

    print("HI")
    return ImageDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        image_size=image_size,
        random_seed=random_seed,
        cuda=cuda,
    )


if __name__ == "__main__":
    app()
