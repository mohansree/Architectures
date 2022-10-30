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
