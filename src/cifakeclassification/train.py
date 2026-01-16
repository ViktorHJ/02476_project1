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


def train(batch_size: int = 128, epochs: int = 10) -> None:
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
        }
    )
    run = wandb_logger.experiment

    datamodule = ImageDataModule(
        batch_size=batch_size,
        num_workers=4,
        val_split=0.2,
    )

    model = Cifake_CNN()

    trainer = pl.Trainer(
        max_epochs=epochs,
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

    run_id = run.id
    run_name = run.name
    with open("reports/last_run.txt", "w") as f:
        f.write("Run name: " + run_name + "\n" + "Run ID: " + run_id)

    wandb.Api()

    wandb.finish()

    # torch.save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    typer.run(train)

# testing workflows
