from cifakeclassification.model import Cifake_CNN
from cifakeclassification.data import ImageDataModule
import torch
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import typer

app = typer.Typer()


@app.command()
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


def main() -> None:
    app()

if __name__ == "__main__":
    main()