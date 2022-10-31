"""Test dense block.

Test the dense block.
    
Usage: 
    pytest tests/test_dence_block.py

"""
from architectures import dense_121_dense_block


def test_dense_block_exist():
    """Assert dense block exist.

    Assert the dense block is exist.

    Args:
        None.

     Returns:
        None.

    """
    assert hasattr(dense_121_dense_block, "DenseBlock")
