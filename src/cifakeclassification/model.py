# ⚡ PyTorch Lightning
import torch
from torch import nn
import pytorch_lightning as pl


def conv_block(in_ch, out_ch, activation_cls, use_bn=True, pool=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(activation_cls())
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """
    Residual block with two 3×3 convolutions and a skip connection.

    Structure
    - conv1 -> BatchNorm -> activation
    - conv2 -> BatchNorm
    - add skip connection (input + conv2 output) -> activation

    Notes
    - Input and output have the same number of channels.
    - Preserves spatial resolution (padding=1, kernel_size=3).
    - Designed to improve gradient flow for deeper nets.
    """

    def __init__(self, channels, activation_cls=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = activation_cls()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.act(out + x)


class Cifake_CNN(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.3,
        optimizer: str = "adam",
        activation_function: str = "relu",
        architecture: str = "CNN_tiny",
    ):
        # Save hyperparameters to W&B automatically
        super().__init__()
        self.save_hyperparameters()

        # Activation classes (store classes, not instances)
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
        }
        activation_cls = activation_map.get(activation_function, nn.ReLU)

        # Architecture selection
        architectures = {
            "CNN_tiny": [16, 32],
            "CNN_small": [32, 64],
            "CNN_medium": [32, 64, 128],
            "CNN_big": [32, 64, 128, 256],
            "CNN_wide": [64, 128],
            "resnet_like": [32, 64, 128, "residual"],
        }

        channels = architectures.get(architecture, architectures["CNN_small"])
        blocks = []
        in_ch = 3
        prev_out = None
        for out in channels:
            if out == "residual":
                if prev_out is None:
                    raise ValueError("residual must follow a numeric channel entry")
                blocks.append(ResidualBlock(prev_out, activation_cls=activation_cls))
                in_ch = prev_out
            else:
                blocks.append(conv_block(in_ch, out, activation_cls=activation_cls))
                prev_out = out
                in_ch = out

        # Use adaptive pooling to get fixed-size features
        self.feature_extractor = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.hparams.dropout_rate)
        self.fc = nn.Linear(in_ch, num_classes)

        optim = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
        self.optimizer_class = optim.get(optimizer, torch.optim.Adam)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def example_input_array(self):
        return torch.randn(1, 3, 32, 32)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    def training_step(self, batch, batch_idx=None):
        img, target = batch
        preds = self(img)
        loss = self.loss_fn(preds, target)
        acc = (preds.argmax(dim=1) == target).float().mean()
        self.log("train/batch_idx", batch_idx, on_step=True, prog_bar=False)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx=None):
        img, target = batch
        preds = self(img)
        loss = self.loss_fn(preds, target)
        acc = (preds.argmax(dim=1) == target).float().mean()
        self.log("val/loss", loss, on_epoch=True)
        self.log("val/acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.hparams.learning_rate)


def test_harness(device="cpu", batch_size=32, architecture="CNN_tiny"):
    print("\n🔍 Running model.py test_harness...", device, architecture)
    bs = batch_size
    model = Cifake_CNN(architecture=architecture)
    model.to(device)

    x = torch.randn(bs, 3, 32, 32, device=device)
    y = torch.randint(0, int(model.hparams.num_classes), (bs,), device=device)

    # Forward pass + shape assertion
    preds = model(x)
    expected = (bs, int(model.hparams.num_classes))
    assert tuple(preds.shape) == expected, f"Expected {expected}, got {tuple(preds.shape)}"
    print("Forward pass shape OK:", tuple(preds.shape))

    # Loss + backward + optimizer step
    loss = model.loss_fn(preds, y)
    optimizer = model.configure_optimizers()
    optimizer.zero_grad()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters()), "No gradients after backward"
    optimizer.step()
    print("Backward and optimizer step OK. Gradients present and optimizer stepped.")
    print("Loss:", loss.item())

    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Optional FLOPs attempt (guarded)
    try:
        from pytorch_lightning.utilities.model_summary import ModelSummary

        summary = ModelSummary(model, max_depth=-1)
        print(f"Estimated FLOPs: {summary.total_flops:,}")
    except Exception:
        print("FLOPs estimation skipped (Lightning summary unavailable).")

    print("\n✅ Model tests completed.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--architecture", default="CNN_tiny")
    args = parser.parse_args()

    # prefer cuda only if available
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    test_harness(device=device, batch_size=args.batch_size, architecture=args.architecture)
