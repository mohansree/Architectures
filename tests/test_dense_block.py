"""Test dense block.

Test the dense block.
    
Usage: 
    pytest tests/test_dence_block.py

"""

import inspect

import pytest
import torch

from architectures import dense_121_dense_block


def test_dense_block_exist():
    """Assert dense block exist.

    Assert the dense block is exist.

    Args:
        None.

     Returns:
        None.

    """
    assert hasattr(dense_121_dense_block, "DenseLayer")


def test_dense_block_is_a_class():
    """Assert dense block is a class.

    Assert dense block is a class.

    Args:
        None.

    Returns:
        None

    """
    assert inspect.isclass(dense_121_dense_block.DenseLayer)


def test_dense_block_is_a_nn_module() -> None:
    """Assert dense block is a module.

    Assert dense block is a module.

    Args:
        None

    Returns:
        None

    """
    assert issubclass(dense_121_dense_block.DenseLayer, torch.nn.Module)


@pytest.mark.parametrize(
    "attribute",
    (
        "batch_norm_1",
        "relu_1",
        "conv_1",
        "batch_norm_2",
        "relu_2",
        "conv_2",
    ),
)
def test_dense_layer_has_attribute(attribute: str) -> None:
    """Assert batchnorm.

    Assert batchnorm layer in DenseLayer.

    Args:
        None

    Returns:
        None

    """
    assert hasattr(dense_121_dense_block.DenseLayer(), attribute)
