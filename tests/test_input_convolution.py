"""Test input convolution layer.

Test the input convolution layer.

Usage:
    python -m pytest Architectures/test_input_convolution
"""

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
