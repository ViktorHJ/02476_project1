import torch
from torch.utils.data import Dataset, DataLoader

from cifakeclassification.data import create_data_module, ImageDataModule


def test_dataset():
    """Test dataset properties."""
    datamodule = create_data_module("data", val_split=0.2)
    datamodule.setup()

    assert datamodule.num_classes == 2
    assert datamodule.classes == ["FAKE", "REAL"]

    val_size = int(100_000 * datamodule.val_split)

    assert len(datamodule.train_dataset) == 100_000 - val_size
    assert len(datamodule.val_dataset) == val_size
    assert len(datamodule.test_dataset) == 20000

    for img, label in datamodule.train_dataset:
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)
        assert label in [0, 1]
        break


def test_class_init():
    """Test the ImageDataModule class."""
    datamodule = create_data_module("data")
    assert isinstance(datamodule, ImageDataModule)
    assert isinstance(datamodule.generator, torch.Generator)


def test_pt_lightning_setup():
    """Test the setup method of the ImageDataModule class."""
    datamodule = create_data_module("data", val_split=0.2)
    datamodule.setup()

    assert isinstance(datamodule.train_dataset, Dataset)
    assert isinstance(datamodule.val_dataset, Dataset)
    assert isinstance(datamodule.test_dataset, Dataset)


def test_dataloaders():
    """Test the dataloader methods of the ImageDataModule class."""
    datamodule = create_data_module("data", batch_size=16)
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    assert train_batch[0].shape == (16, 3, 32, 32)
    assert val_batch[0].shape == (16, 3, 32, 32)
    assert test_batch[0].shape == (16, 3, 32, 32)
