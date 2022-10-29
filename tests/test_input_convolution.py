"""Test input convolution layer.

Test the input convolution layer.

Usage:
    python -m pytest Architectures/test_input_convolution
"""

import torch

from architectures import densenet_121_input_convolution_layer


def test_input_convolution_exist() -> None:
    """Test the input convolution exist.

    Test the input convolution layer exist.

    Args:
        None

    Returns:
        None

    """
    assert hasattr(densenet_121_input_convolution_layer, "InputConvolutionLayer")


def test_input_convolution_subclass_nn_module() -> None:
    """Assert InputConvolutionLayer is torch module.

    Assert InputConvolutionLayer is a torch module.

    Args:
        None

    Returns:
        None

    """
    assert issubclass(
        densenet_121_input_convolution_layer.InputConvolutionLayer, torch.nn.Module
    )
