from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from cifakeclassification.model import Cifake_CNN
from cifakeclassification.data import ImageDataModule

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"  # -> 02476_project1/configs
@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")



# @hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    hp = cfg.hyperparameters

    datamodule = ImageDataModule(
        batch_size=hp.batch_size,
        num_workers=0,
        val_split=0.2,
    )

    model = Cifake_CNN(
        activation_function=hp.activation_function,
        dropout_rate=hp.dropout_rate,
        learning_rate=hp.learning_rate,
        optimizer=hp.optimizer,
    )

    trainer = pl.Trainer(
        max_epochs=hp.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=datamodule)

    out_dir = Path(hydra.utils.to_absolute_path("models"))
    out_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pth")

if __name__ == "__main__":
    train()
