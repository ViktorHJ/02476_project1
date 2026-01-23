"""
Training script for the CIFAKE classification project.

This script integrates:
- Hydra for configuration management
- PyTorch Lightning for structured training
- Weights & Biases for experiment tracking
- Automatic hardware-aware precision selection
- Model complexity logging (parameters + FLOPs)
- Environment diagnostics for reproducibility

The script loads configuration from the `configs/` directory,
initializes the data module and model, logs metadata to W&B,
runs training, saves the final checkpoint, and logs it as an artifact.
"""

from pathlib import Path
import os

# ⚕️ Hydra
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

# Project modules
from cifakeclassification.model import Cifake_CNN
from cifakeclassification.data import ImageDataModule

# Environment variable loader
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Global initialization: seeding, environment variables, backend settings
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
pl.seed_everything(hash("Random seed:") % 2**32 - 1, workers=True)
load_dotenv()

# GPU backend
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True


def auto_precision():
    """
    Automatically selects the optimal mixed-precision mode based on hardware.

    Returns:
        str: A PyTorch Lightning precision string:
            - "bf16-mixed" on Apple Silicon (M1/M2/M3) (best supported)
            - "16-mixed" on NVIDIA GPUs
            - "32-true" on CPU fallback
    """
    if torch.backends.mps.is_available():
        return "bf16-mixed"
    if torch.cuda.is_available():
        return "16-mixed"
    return "32-true"


# ---------------------------------------------------------------------------
# Main training function (Hydra entry point)
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def train(cfg: DictConfig):
    """
    Main training pipeline.
    Args:
        cfg (DictConfig): Hydra configuration object containing:
            - hyperparameters
            - model settings
            - data settings
            - training settings
    """
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

    # --- Dynamic Workers ---
    if torch.backends.mps.is_available():
        num_workers = 0  # Default Mac:
    elif torch.cuda.is_available():
        num_workers = torch.cuda.device_count() * 4  # Default CUDA:
    else:
        # WSL / 4-core 8-thread:
        num_workers = min(4, os.cpu_count() or 1)

    # --- Data Module ---
    datamodule = ImageDataModule(
        batch_size=hp.batch_size,
        num_workers=num_workers,
        val_split=0.2,
    )

    # --- Model initialization ---
    model = Cifake_CNN(
        learning_rate=hp.learning_rate,
        dropout_rate=hp.dropout_rate,
        optimizer=hp.optimizer,
        activation_function=hp.activation_function,
        architecture=hp.architecture,
    )

    # --- Model complexity logging ---
    num_params = sum(p.numel() for p in model.parameters())

    try:
        from pytorch_lightning.utilities.model_summary import ModelSummary

        summary = ModelSummary(model, max_depth=-1)
        flops = summary.total_flops
    except Exception:
        flops = None

    wandb_logger.log_hyperparams(
        {
            "complexity/params": num_params,
            "complexity/flops": flops,
        }
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=hp.max_epochs,
        log_every_n_steps=50,
        # Disable model summary to avoid redundant logging and implement manual full logging, change flag from -1 to 0,1,2... to limit depth
        enable_model_summary=False,
        callbacks=[SummaryCallback(max_depth=-1)],
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision=auto_precision(),
        logger=wandb_logger,
    )

    # -----------------------------------------------------------------------
    # Runtime diagnostics
    # -----------------------------------------------------------------------
    print("\n=== Runtime Diagnostics ===")
    # CUDA / MPS diagnostics
    print(f"[CUDA] Available: {torch.cuda.is_available()}  |  [MPS] Available: {torch.backends.mps.is_available()}")
    # cuDNN diagnostics
    print(f"[cuDNN] Enabled: {torch.backends.cudnn.enabled}  |  [cuDNN] Version: {torch.backends.cudnn.version()}")
    # Precision diagnostics
    print(
        f"[Precision] Selected: {trainer.precision}  |  "
        f"[Matmul] float32 precision: {torch.get_float32_matmul_precision()}"
    )
    print("===========================")

    # --- Training ---
    trainer.fit(model, datamodule=datamodule)

    # --- Save model ---
    out_dir = Path(hydra.utils.to_absolute_path("models"))
    # save_path = "models/model.pth"
    out_dir.mkdir(exist_ok=True)
    save_path = f"{out_dir}/model.ckpt"
    trainer.save_checkpoint(save_path, weights_only=False)

    # --- Log artifact ---
    artifact = wandb.Artifact("cifake-model", type="model")
    artifact.add_file(save_path)
    wandb_logger.experiment.log_artifact(artifact)

    wandb.finish()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
