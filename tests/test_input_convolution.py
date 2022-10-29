"""Test input convolution layer.

Test the input convolution layer.

Usage:
    python -m pytest Architectures/test_input_convolution
"""

import architectures


def test_input_convolution_exist() -> None:
    """Test the input convolution exist.

    Test the input convolution layer exist.

    Args:
        None

    Returns:
        None

    """
    assert hasattr(
        architectures.densenet_121_input_convolution_layer, "InputConvolutionLayer"
    )
