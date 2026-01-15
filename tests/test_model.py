from src.cifakeclassification.model import Cifake_CNN
import torch


def test_model_shape():
    model = Cifake_CNN()
    x = torch.rand(2, 1, 28, 28)
    assert model(x).shape == torch.Size([2,10])
