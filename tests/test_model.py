from cifakeclassification.model import Cifake_CNN
import torch


def test_model_shape():
    model = Cifake_CNN()
    x = torch.rand(2, 3, 32, 32)
    assert model(x).shape == torch.Size([2, 2])
