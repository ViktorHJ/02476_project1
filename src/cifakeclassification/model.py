"""Model definitions for CIFAKE classification.

Includes:
- A flexible CNN architecture family (tiny → big)
- Optional residual blocks
- Clean activation/optimizer mapping
- A test harness for quick local validation
"""

# ⚡ PyTorch Lightning
import torch
from torch import nn
import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# Building blocks for CNN architectures
# ---------------------------------------------------------------------------
def conv_block(in_ch, out_ch, activation_cls, use_bn=True, pool=True):
    """Standard Conv → BN → Activation → Optional MaxPool block."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(activation_cls())
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions and a skip connection.

    Structure
    - conv → bn → act → conv → bn
    - → skip-add (input + conv2 output) -> act

    Notes
    - Input/output channels must match.
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
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


# ---------------------------------------------------------------------------
# Lightning model
# ---------------------------------------------------------------------------
class Cifake_CNN(pl.LightningModule):
    """
    Flexible parameterized CNN classifier for CIFAKE.

    Supports:
    - Multiple architecture selections
    - Custom activation functions
    - Optional residual blocks
    - Automatic hyperparameter logging via Lightning
    """

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

        # Activation mapping
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

        if architecture not in architectures:
            raise ValueError(f"Unknown architecture '{architecture}'. Valid options: {list(architectures.keys())}")
        channels = architectures[architecture]

        # Build feature extractor
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

        # Optimizer mapping
        optim = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
        self.optimizer_class = optim.get(optimizer, torch.optim.Adam)
        self.loss_fn = nn.CrossEntropyLoss()

    # Lightning uses this for model summary & FLOPs
    @property
    def example_input_array(self):
        return torch.randn(1, 3, 32, 32)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    # -----------------------------------------------------------------------
    # Training / Validation / Test steps
    # -----------------------------------------------------------------------
    def training_step(self, batch, batch_idx=None):
        img, target = batch
        preds = self(img)
        loss = self.loss_fn(preds, target)
        acc = (preds.argmax(dim=1) == target).float().mean()

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

    def test_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.loss_fn(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.hparams.learning_rate)


# ---------------------------------------------------------------------------
# Test harness (local debugging)
# ---------------------------------------------------------------------------
def test_harness(device="cpu", batch_size=32, architecture="CNN_tiny"):
    """
    Quick local test to verify:
    - Forward pass shape
    - Backward pass + gradients
    - Optimizer step
    - Parameter count
    - Optional FLOPs estimation
    """
    print("\n🔍 Running model.py test_harness...", device, architecture)

    model = Cifake_CNN(architecture=architecture).to(device)
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, int(model.hparams.num_classes), (batch_size,), device=device)

    # Forward pass + shape assertion
    preds = model(x)
    expected = (batch_size, int(model.hparams.num_classes))
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
        print("FLOPs estimation unavailable.")

    print("\n✅ Model tests completed.\n")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
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
