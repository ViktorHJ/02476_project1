from cifakeclassification.model import Cifake_CNN
from cifakeclassification.data import ImageDataModule

import pytorch_lightning as pl
import typer


def evaluate(model_checkpoint: str) -> None:
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    # Load model from checkpoint
    model = Cifake_CNN.load_from_checkpoint(model_checkpoint)

    # Data
    datamodule = ImageDataModule()
    datamodule.setup()
    test_dataloader = datamodule.test_dataloader()

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,  # no logging needed here
    )

    # Run validation
    results = trainer.validate(model, dataloaders=test_dataloader)

    print(results)


if __name__ == "__main__":
    typer.run(evaluate)
