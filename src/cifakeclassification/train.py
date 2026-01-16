from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from cifakeclassification.model import Cifake_CNN
from cifakeclassification.data import ImageDataModule
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv
import wandb
import os


load_dotenv()


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    hp = cfg.hyperparameters

    # W&B config from .env
    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")

    wandb_logger = WandbLogger(
        project=project,
        entity=entity,
        log_model=True,
        save_dir=".",
    )

    # Log hyperparameters to W&B
    wandb_logger.experiment.config.update(
        {
            "batch_size": hp.batch_size,
            "epochs": hp.epochs,
            "learning_rate": hp.learning_rate,
            "dropout_rate": hp.dropout_rate,
            "optimizer": hp.optimizer,
            "activation_function": hp.activation_function,
            "architecture": hp.architecture,
        }
    )

    # Data
    datamodule = ImageDataModule(
        batch_size=hp.batch_size,
        num_workers=4,
        val_split=0.2,
    )

    # Model
    model = Cifake_CNN(
        learning_rate=hp.learning_rate,
        dropout_rate=hp.dropout_rate,
        optimizer=hp.optimizer,
        activation_function=hp.activation_function,
        architecture=hp.architecture,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=hp.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=datamodule)

    # Save model
    save_path = "models/model.pth"
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)

    # Log artifact
    artifact = wandb.Artifact("cifake-model", type="model")
    artifact.add_file(save_path)
    wandb_logger.experiment.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    train()
