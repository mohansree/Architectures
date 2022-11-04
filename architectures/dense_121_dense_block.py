"""
The dense block.

Implementing the Dense Block.

Usage:
    from architectures.dense_121_dense_block import DenseBlock
"""

import torch


class DenseLayer(torch.nn.Module):  # dead: disable
    """DenseBlock.

    The dense block.

    Attributes:
        None

    """

    def __init__(
        self,
    ) -> None:
        """Construct DenseLayer.

        Constructs DenseLayer.

        Args:
            None

        Returns:
            None

        """
        super().__init__()
        self.batch_norm_1: None = torch.nn.BatchNorm2d(
            1,
        )
        self.relu_1: None = None
        self.conv_1: None = None
        self.batch_norm_2: None = None
        self.relu_2: None = None
        self.conv_2: None = None
