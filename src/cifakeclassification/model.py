from omegaconf import DictConfig
from torch import nn
import torch
import pytorch_lightning as pl
import hydra
from pathlib import Path


class Cifake_CNN(pl.LightningModule):
    def __init__(
        self,
        activation_function: str = "relu",
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()

        activation_functions = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }

        optim = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }

        self.optimizer = optim[optimizer]

        self.activation = activation_functions[activation_function]
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def training_step(self, batch):
        """Training step."""
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)

        acc = (y_pred.argmax(dim=1) == target).float().mean()

        # Lightning handles aggregation + logging
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.loss_fn(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def test_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.loss_fn(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# @hydra.main(version_base=None, config_path="../../configs", config_name="config")
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"  # -> 02476_project1/configs


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    hp = cfg.hyperparameters

    model = Cifake_CNN(
        activation_function=hp.activation_function,
        dropout_rate=hp.dropout_rate,
        learning_rate=hp.learning_rate,
        optimizer=hp.optimizer,
    )

    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
    print(f"Hyperparameters: {hp}")


if __name__ == "__main__":
    main()
