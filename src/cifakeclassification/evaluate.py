import os
from pathlib import Path

import hydra
import torch
import wandb
import pytorch_lightning as pl
from omegaconf import DictConfig

from dotenv import load_dotenv

from cifakeclassification.model import Cifake_CNN
from cifakeclassification.data import ImageDataModule

load_dotenv()

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"  # -> 02476_project1/configs


def log_test_predictions(model, dataloader, run, num_samples=32):
    """
    Logs a sample of test images and model predictions to a W&B Table.
    """
    columns = ["image", "prediction", "ground_truth", "score_real", "score_fake"]
    test_table = wandb.Table(columns=columns)

    model.eval()
    # Get one batch of data
    batch = next(iter(dataloader))
    imgs, labels = batch

    with torch.no_grad():
        logits = model(imgs)
        preds = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

    # Class labels for CIFAKE
    class_labels = ["Real", "Fake"]

    for i in range(min(num_samples, len(imgs))):
        test_table.add_data(
            wandb.Image(imgs[i]),  # Image
            class_labels[preds[i]],  # Model prediction
            class_labels[labels[i]],  # Ground truth
            float(probs[i][0]),  # Probability of Real
            float(probs[i][1]),  # Probability of Fake
        )

    run.log({"test_visual_results": test_table})


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="evaluation_config")
def evaluate(cfg: DictConfig) -> None:
    models_dir = "models/"

    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")

    run = wandb.init(
        project=project,
        entity=entity,
        job_type="evaluation",  # Labels this as an eval run, not training
        config=dict(cfg),  # Keep track of what data settings were used
    )

    # Load model from checkpoint
    print(f"--- Evaluating model from: {cfg.model.model_checkpoint} ---")
    file_path = Path(models_dir + cfg.model.model_checkpoint)
    if file_path.exists():
        model = Cifake_CNN.load_from_checkpoint(models_dir + cfg.model.model_checkpoint)
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {file_path}")

    # Setup Data
    datamodule = ImageDataModule(
        data_dir=cfg.data.directory,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        random_seed=cfg.data.random_seed,
    )
    datamodule.setup(stage="test")

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,  # no logging needed here
    )

    # Run validation
    results = trainer.test(model, datamodule=datamodule)

    # Log results to W&B
    if results:
        wandb.log(results[0])

    print("Generating Visual Tables for W&B...")
    log_test_predictions(model, datamodule.test_dataloader(), run)

    run.finish()

    print("Test Results:", results[0] if results else "No results")
    print("Evaluation Complete.")


if __name__ == "__main__":
    evaluate()
