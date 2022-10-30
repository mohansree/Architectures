"""Test the pooling layer.

Tests for the pooling layer.

Usage:
    python -m pytest -m tests/test_pooling_layer
"""

import torch

from architectures import densenet_121_pooling_layer


def test_pooling_layer_exist() -> None:
    """Test the pooling layer exist.

    Test the pooling layer exist.

    Args:
        None

    Returns:
        None

    """
    assert hasattr(densenet_121_pooling_layer, "PoolingLayer")


def test_pooling_layer_is_nn_module() -> None:
    """Test pooling layer subclasses nn.Module.

    Assert pooling layer subclasses nn.Module.

    Args:
        None

    Returns:
        None

    """
    assert issubclass(densenet_121_pooling_layer.PoolingLayer, torch.nn.Module)


def test_pooling_layer_has_attribute_layer() -> None:
    """Assert pooling layer exist.

    Assert pooling layer exist.

    Args:
        None

    Returns:
        None

    """
    assert hasattr(densenet_121_pooling_layer.PoolingLayer(), "layer")


def test_layer_is_pooling_layer() -> None:
    """Assert layer is a pooling layer.

    Assert layer is a pooling layer.

    Args:
        None

    Returns:
        None

    """
    assert isinstance(
        densenet_121_pooling_layer.PoolingLayer().layer, torch.nn.MaxPool2d
    )


def test_output_shape_is_correct() -> None:
    """Assert output shape is correct.

    Assert output shape is correct.

    Args:
        None

    Returns:
        None

    """
    input_tensor: torch.Tensor = torch.zeros(
        (
            3,
            112,
            112,
        )
    )
    layer: torch.nn.Module = densenet_121_pooling_layer.PoolingLayer(
        kernel_size=3,
        stride=2,
    )
    assert layer(input_tensor).shape == (
        3,
        56,
        56,
    )
