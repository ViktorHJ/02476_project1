from pathlib import Path

import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig

from cifakeclassification.model import Cifake_CNN
from cifakeclassification.data import ImageDataModule

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"  # -> 02476_project1/configs


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="data_evaluate")
def evaluate(cfg: DictConfig) -> None:
    models_dir = "models/"

    # Load model from checkpoint
    model = Cifake_CNN.load_from_checkpoint(models_dir + cfg.data.model)
    # Data
    datamodule = ImageDataModule(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        random_seed=cfg.data.random_seed,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,  # no logging needed here
    )

    # Run validation
    results = trainer.test(model, datamodule=datamodule)

    print(results)


if __name__ == "__main__":
    evaluate()
