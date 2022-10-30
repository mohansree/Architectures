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


def test_input_convolution_layer_has_attribute_layer() -> None:
    """Assert convolution layer exist.

    Assert convolution layer exist.

    Args:
        None

    Returns:
        None

    """
    assert hasattr(
        densenet_121_input_convolution_layer.InputConvolutionLayer(), "layer"
    )


def test_layer_is_convolution() -> None:
    """Assert layer is convolution.

    Assert the layer is a convolution layer.

    Args:
        None

    Returns:
        None

    """
    assert isinstance(
        densenet_121_input_convolution_layer.InputConvolutionLayer().layer,
        torch.nn.Conv2d,
    )


def test_InputConvolutionLayer_accept_arguments() -> None:
    """Assert the module accept arguments.

    Assert the module accept arguments.

    Args:
        None

    Returns:
        None

    """
    conv = densenet_121_input_convolution_layer.InputConvolutionLayer(
        in_channels=3,
        out_channels=3,
        kernel_size=7,
        stride=2,
    )
    assert conv.layer.in_channels == 3
    assert conv.layer.out_channels == 3
    assert conv.layer.kernel_size == (7, 7)
    assert conv.layer.stride == (
        2,
        2,
    )


def test_output_size() -> None:
    """Assert InputConvolutionLayer output size.

    Assert InputConvolutionLayer returns correct shape.

    Args:
        None

    Returns:
        None

    """
    conv = densenet_121_input_convolution_layer.InputConvolutionLayer(
        in_channels=3,
        out_channels=3,
        kernel_size=7,
        stride=2,
    )
    input_tensor: torch.Tensor = torch.zeros((3, 224, 224))
    assert conv(input_tensor).shape == (3, 112, 112)
