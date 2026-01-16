from torch import nn
import torch
import pytorch_lightning as pl

class Cifake_CNN(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

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

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def main() -> None:
    model = Cifake_CNN()
    x = torch.rand(2, 3, 32, 32)
    print(model(x).shape)

if __name__ == "__main__":
    main()