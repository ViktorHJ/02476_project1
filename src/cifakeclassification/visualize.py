from pathlib import Path
from typing import Annotated
import os
import typer

from dotenv import load_dotenv

import matplotlib

matplotlib.use("Agg")

import wandb
import matplotlib.pyplot as plt
import pandas as pd

load_dotenv()
app = typer.Typer()


@app.command()
def visualize_train_from_wandb(
    run_name: Annotated[str, typer.Argument(help="The name of the W&B run")],
    output_dir: Annotated[str, typer.Option(help="Where to save figures")] = "reports/figures",
):
    """
    Visualize metrics from a W&B run given its name.
    """
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    # Find run by name
    run = next((r for r in runs if r.name == run_name), None)
    if run is None:
        raise ValueError(f"No run found with name '{run_name}'")

    print(f"Found run: {run.name} (id={run.id})")

    # Get history as DataFrame
    history = pd.DataFrame(run.history(samples=10000))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("train/loss", "Train Loss", "train_loss"),
        ("train/acc", "Train Accuracy", "train_acc"),
        ("val/loss", "Validation Loss", "val_loss"),
        ("val/acc", "Validation Accuracy", "val_acc"),
    ]

    for metric, metric_title, metric_name in metrics:
        if metric not in history:
            print(f"Skipping {metric} (not found)")
            continue

        plt.figure()
        plt.plot(history["_step"][history[metric].notnull()], history[metric][history[metric].notnull()], label=metric)
        plt.xlabel("Steps")
        plt.ylabel(metric_title)
        plt.title(metric_title)
        plt.grid(True)

        save_path = output_path / f"{run.name}_{metric_name}.png"
        plt.savefig(save_path)
        plt.close()

        print(f"Saved {save_path}")


@app.command()
def visualize_eval():
    pass


if __name__ == "__main__":
    app()
