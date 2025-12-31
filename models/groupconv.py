import torch
import torch.nn as nn
from torchinfo import summary
import torch
import torch.nn.functional as F
import util.params as params
from models.metrics import *


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, L = x.size()  # L now represents the length of the sequence (instead of H, W)
        g = self.groups
        # Ensure the channels are divisible by groups
        assert C % g == 0, "The number of channels must be divisible by groups."
        x = x.view(N, g, C // g, L)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(N, C, L)
        return x


class GroupConvShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(GroupConvShuffle, self).__init__()
        self.group_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    groups=groups, bias=False)
        self.channel_shuffle = ChannelShuffle(groups)

    def forward(self, x):
        x = self.group_conv(x)
        x = self.channel_shuffle(x)
        return x
