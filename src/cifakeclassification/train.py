from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from cifakeclassification.model import Cifake_CNN
from cifakeclassification.data import ImageDataModule
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv
import torch
import pytorch_lightning as pl
import typer
import wandb
import os


load_dotenv()


def train(
    batch_size: int = 128,
    epochs: int = 2,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    optimizer: str = "adam",
    activation_function: str = "relu",
    architecture: str = "Cifake_CNN_small",
):
    """Train a model on MNIST with W&B."""

    # W&B config from .env
    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")

    wandb_logger = WandbLogger(
        project=project,
        entity=entity,
        log_model=True,
        save_dir=".",
    )

    wandb_logger.experiment.config.update(
        {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "optimizer": optimizer,
            "activation_function": activation_function,
            "architecture": architecture,
        }
    )

    datamodule = ImageDataModule(
        batch_size=hp.batch_size,
        num_workers=0,
        val_split=0.2,
    )

    model = Cifake_CNN(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        optimizer=optimizer,
        activation_function=activation_function,
        architecture=architecture,
    )

    trainer = pl.Trainer(
        max_epochs=hp.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=datamodule)

    # -------------------------
    # Save + log artifact
    # -------------------------
    save_path = "models/model.pth"
    torch.save(model.state_dict(), save_path)

    artifact = wandb.Artifact("cifake-model", type="model")
    artifact.add_file(save_path)
    wandb_logger.experiment.log_artifact(artifact)

    wandb.finish()

    # torch.save(model.state_dict(), "models/model.pth")

    out_dir = Path(hydra.utils.to_absolute_path("models"))
    out_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pth")

if __name__ == "__main__":
    train()
