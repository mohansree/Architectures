"""Test the pooling layer.

Tests for the pooling layer.

Usage:
    python -m pytest -m tests/test_pooling_layer
"""

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
