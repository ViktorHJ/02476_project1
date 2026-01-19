from cifakeclassification.model import Cifake_CNN
from cifakeclassification.data import ImageDataModule
from dotenv import load_dotenv
import pytorch_lightning as pl
import typer
import wandb
import os

load_dotenv()


def evaluate(model_checkpoint: str) -> None:
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")

    with wandb.init(project=project, entity=entity, job_type="evaluation") as run:
        # Load model once
        model = Cifake_CNN.load_from_checkpoint(model_checkpoint)

        # Data
        datamodule = ImageDataModule()
        datamodule.setup()
        test_dataloader = datamodule.test_dataloader()

        # Trainer
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            logger=False,
        )

        # Run validation
        results = trainer.validate(model, dataloaders=test_dataloader)

        # Log metrics to W&B
        run.log(
            {
                "test/loss": results[0]["val_loss"],
            }
        )

        print(results)


if __name__ == "__main__":
    typer.run(evaluate)
