from cifakeclassification.model import Cifake_CNN
from cifakeclassification.data import ImageDataModule
import torch
import pytorch_lightning as pl
import typer


def train(batch_size: int = 128, epochs: int = 10) -> None:
    """Train a model on MNIST."""

    datamodule = ImageDataModule(
        batch_size=batch_size,
        num_workers=0,
        val_split=0.2,
    )

    model = Cifake_CNN()

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=datamodule)
    torch.save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    typer.run(train)

# testing workflows
