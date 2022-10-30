"""Pooling layer.

Implements pooling layer.

Usage:
    from architectures.densenet_121_pooling_layer import PoolingLayer
"""

import torch


class PoolingLayer(torch.nn.Module):  # dead: disable
    """Pooling layer.

    Implement pooling layer.

    Attributes:
        None

    """

    def __init__(
        self,
        kernel_size=None,
        stride=None,
    ) -> None:
        """Construct PoolingLayer.

        Constructs PoolingLayer.

        Args:
            None

        Returns:
            None

        """
        super().__init__()
        self.layer = torch.nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(  # dead: disable
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Implement forward.

        Pass through pooling layer.

        Args:
            features: Input tensor.

        Returns:
            Result from layer.

        """
        return self.layer(features)
