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
        in_channels=None,
        out_channels=None,
        kernel_size=None,
        stride=None,
    ) -> None:
        """Construct InputConvolutionLayer.

        Construct the InputConvolutionLayer.

        Args:
            None

        Returns:
            None

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer = torch.nn.Conv2d(
            in_channels,
            out_channels,
            1,
        )


if __name__ == "__main__":
    InputConvolutionLayer()
