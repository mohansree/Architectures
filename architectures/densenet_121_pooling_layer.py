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
    ) -> None:
        """Construct PoolingLayer.

        Constructs PoolingLayer.

        Args:
            None

        Returns:
            None

        """
        self.layer = torch.nn.MaxPool2d()
