from cifakeclassification.model import Cifake_CNN
import torch

def test_model_forward_pass():
    """
    Test to verify the forward pass of the Cifake_CNN model.
    """
    print("\n🔍 Running model forward pass test...")

    # Create a dummy input tensor with batch size 4 and image size 3x32x32
    dummy_input = torch.randn(4, 3, 32, 32)

    # Initialize the model
    model = Cifake_CNN()

    # Perform a forward pass
    output = model(dummy_input)

    # Check output shape
    assert output.shape == (4, 2), f"Expected output shape (4, 2), but got {output.shape}"

