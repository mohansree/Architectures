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
        in_features: int = 1,
    ) -> None:
        """Construct DenseLayer.

        Constructs DenseLayer.

        Args:
            None

        Returns:
            None

        """
        super().__init__()
        self.batch_norm: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(
            in_features,
        )
