"""Construct densenet_121_input_convolution_layer.

Construct densenet_121_input_convolution_layer.

Usage:
    Implicit usage.
"""

import torch


class InputConvolutionLayer(torch.nn.Module):
    """Input convolution layer.

    The input convolution layer.
    """

    def __init__(
        self,
    ) -> None:
        """Construct InputConvolutionLayer.

        Construct the InputConvolutionLayer.

        Args:
            None

        Returns:
            None

        """
        super().__init__()
        self.layer = torch.nn.Conv2d(
            1,
            1,
            1,
        )


if __name__ == "__main__":
    InputConvolutionLayer()
