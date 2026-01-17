from pathlib import Path
import hydra
from omegaconf import DictConfig

# 🍦 Vanilla PyTorch
import torch

# ⚡ PyTorch Lightning
import pytorch_lightning as pl

# 🏋️‍♀️ Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary as SummaryCallback


from cifakeclassification.model import Cifake_CNN
from cifakeclassification.data import ImageDataModule

from dotenv import load_dotenv
import os

pl.seed_everything(hash("setting random seeds") % 2**32 - 1)
load_dotenv()

# Add this here to ensure it is set before the Trainer starts
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True


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
            "max_epochs": hp.max_epochs,
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
        num_workers=8,
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

    # FLOPs + params
    # summary = SummaryUtility(model, max_depth=-1)
    # wandb.log({"model/FLOPs": summary.total_flops})
    # wandb.log({"model/num_params": sum(p.numel() for p in model.parameters())})

    # Trainer
    trainer = pl.Trainer(
        max_epochs=hp.max_epochs,
        log_every_n_steps=50,
        enable_model_summary=True,
        callbacks=[SummaryCallback(max_depth=-1)],
        accelerator="auto",
        devices="auto",
        precision="16-mixed",  # safe: ignored on CPU
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
