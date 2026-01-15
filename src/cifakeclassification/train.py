from cifakeclassification.model import Cifake_CNN
# from data import MyDataset
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
    print("Training day and night")
    # Data
    # train_set, _ = MyDataset
    # train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    train_dataset = MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    model = Cifake_CNN()

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",  # uses CUDA/MPS/CPU automatically
        devices="auto",
        log_every_n_steps=50,
    )

    trainer.fit(model, train_dataloaders=train_dataloader)
    torch.save(model.state_dict(), "models/model.pth")


def main() -> None:
    app()

if __name__ == "__main__":
    main()