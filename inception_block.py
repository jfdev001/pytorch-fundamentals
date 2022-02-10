"""Implements inception block based on PyTorch Tutorial 4 Figure.

https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/04-inception-resnet-densenet.html

NOTE: Batch norm after each convolution, then activation function.
"""

from typing import List

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """Inception block before looking at pytorch solution.

    Think of branches flowing in columns, not as rows. Since the computation
    graph flows vertically rather than horizontally, it doesn't make sense
    to separate the layers the way I have.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int, **kwargs):
        super().__init__(**kwargs)

        self.left_one_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)

        self.row_one_left_one_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)

        self.row_one_center_one_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)

        self.max_pool = nn.MaxPool2d(kernel_size=3)

        self.row_two_left_three_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3)

        self.row_two_center_five_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5)

        self.row_two_right_one_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:

        left_one_conv = self.left_one_conv(x)

        # First row computations
        row_one_left_one_conv = self.row_one_left_one_conv(x)
        row_one_center_one_conv = self.row_one_center_one_conv(x)
        max_pool = self.max_pool(x)

        # Second row outputs
        row_two_left_three_conv = self.row_two_left_three_conv(
            row_one_left_one_conv)
        row_two_center_five_conv = self.row_two_center_five_conv(
            row_one_center_one_conv)
        row_two_right_one_conv = self.row_two_right_one_conv(max_pool)

        # Concatenate outputs from second row and left convolution
        filter_concat = torch.cat(
            (left_one_conv,
             row_two_left_three_conv,
             row_two_center_five_conv,
             row_two_right_one_conv), dim=-1)

        return filter_concat
