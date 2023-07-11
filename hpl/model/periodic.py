import torch.nn as nn
import torch.nn.functional as F


class PeriodicConv1d(nn.Conv1d):
    """1D convolutional layer with circular padding."""

    def forward(self, input):
        if self.padding_mode == "circular":
            expanded_padding_circ = (self.padding[0] // 2, (self.padding[0] - 1) // 2)
            return F.conv1d(
                F.pad(input, expanded_padding_circ, mode="circular"),
                self.weight,
                self.bias,
                self.stride,
                (0,),
                self.dilation,
                self.groups,
            )
        elif self.padding_mode == "valid":
            expanded_padding_circ = (self.padding[0] // 2, (self.padding[0] - 1) // 2)
            return F.conv1d(
                F.pad(input, expanded_padding_circ, mode="constant", value=0.0),
                self.weight,
                self.bias,
                self.stride,
                (0,),
                self.dilation,
                self.groups,
            )
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PeriodicConv2d(nn.Conv2d):
    """2D convolutional layer with mixed zero and circular padding.

    Uses circular padding along last axis (W) and zero-padding on second-last axis (H)
    """

    def conv2d_forward(self, input, weight):
        if self.padding_mode == "circular":
            expanded_padding_circ = ((self.padding[0] + 1) // 2, self.padding[0] // 2, 0, 0)
            expanded_padding_zero = (0, 0, (self.padding[1] + 1) // 2, self.padding[1] // 2)
            return F.conv2d(
                F.pad(F.pad(input, expanded_padding_circ, mode="circular"), expanded_padding_zero, mode="constant"),
                weight,
                self.bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
