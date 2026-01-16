from cifakeclassification.model import Cifake_CNN
import torch


def test_forward_shape():
    model = Cifake_CNN()
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 2)


def test_model_hyperparameters():
    model = Cifake_CNN(
        activation_function="relu",
        dropout_rate=0.3,
        learning_rate=0.001,
        optimizer="adam",
    )

    assert isinstance(model.activation, torch.nn.ReLU)
    assert model.dropout.p == 0.3
    assert model.learning_rate == 0.001
    assert model.optimizer is torch.optim.Adam