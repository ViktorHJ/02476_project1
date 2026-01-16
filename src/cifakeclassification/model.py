from omegaconf import DictConfig
from torch import nn
import torch
import pytorch_lightning as pl
import hydra
from pathlib import Path


class Cifake_CNN(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.3,
        optimizer: str = "adam",
        activation_function: str = "relu",
        architecture: str = "Cifake_CNN_small",
    ):
        super().__init__()

        # Save hyperparameters to W&B automatically
        self.save_hyperparameters()

        # Activation selection
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        self.activation = activations[activation_function]
        self.learning_rate = learning_rate
        # Architecture selection
        arcs = {
            "Cifake_CNN_small": [32, 64],
            "Cifake_CNN_medium": [32, 64, 128],
            "Cifake_Wild_large": [32, 64, 128, 256],
        }

        channels = arcs[architecture]

        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(self.activation)
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)

        # Compute flattened size dynamically
        dummy = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            flat_dim = self.feature_extractor(dummy).numel()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(flat_dim, 2)

        optim = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }
        self.optimizer_class = optim[optimizer]

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    def training_step(self, batch):
        img, target = batch
        preds = self(img)
        loss = self.loss_fn(preds, target)
        acc = (preds.argmax(dim=1) == target).float().mean()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch):
        img, target = batch
        preds = self(img)
        loss = self.loss_fn(preds, target)
        acc = (preds.argmax(dim=1) == target).float().mean()

        self.log("val/loss", loss, on_epoch=True)
        self.log("val/acc", acc, on_epoch=True)

    def test_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.loss_fn(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.hparams.learning_rate)


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"  # -> 02476_project1/configs


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    hp = cfg.hyperparameters

    model = Cifake_CNN(
        activation_function=hp.activation_function,
        dropout_rate=hp.dropout_rate,
        learning_rate=hp.learning_rate,
        optimizer=hp.optimizer,
        architecture=hp.architecture,
    )

    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
    print(f"Hyperparameters: {hp}")


if __name__ == "__main__":
    main()
