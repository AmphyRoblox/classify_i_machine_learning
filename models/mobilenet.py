"""mobilenet in pytorch



[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
"""

import torch
import torch.nn as nn
from torchinfo import summary
import util.params as params
from models.metrics import *


class DepthSeperabelConv1d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                (1, kernel_size),
                groups=input_channels,
                **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicConv1d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        # self.conv = nn.Conv2d(
        #     input_channels, output_channels, (1, kernel_size), **kwargs)
        # self.bn = nn.BatchNorm2d(output_channels)
        # self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=(2, 7), stride=2, padding=(0, 3),
                              bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MobileNet(nn.Module):
    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, num_classes=10, backbone=False):
        super().__init__()

        alpha = width_multiplier
        self.stem = nn.Sequential(
            BasicConv1d(1, int(32 * alpha), 7, padding=(0, 1), bias=False),
            DepthSeperabelConv1d(
                int(32 * alpha),
                int(64 * alpha),
                3,
                padding=(0, 1),
                bias=False
            )
        )

        # downsample
        self.conv1 = nn.Sequential(
            DepthSeperabelConv1d(
                int(64 * alpha),
                int(128 * alpha),
                3,
                stride=2,
                padding=(0, 1),
                bias=False
            ),
            DepthSeperabelConv1d(
                int(128 * alpha),
                int(128 * alpha),
                3,
                padding=(0, 1),
                bias=False
            )
        )

        # downsample
        self.conv2 = nn.Sequential(
            DepthSeperabelConv1d(
                int(128 * alpha),
                int(256 * alpha),
                3,
                stride=2,
                padding=(0, 1),
                bias=False
            ),
            DepthSeperabelConv1d(
                int(256 * alpha),
                int(256 * alpha),
                3,
                padding=(0, 1),
                bias=False
            )
        )

        # downsample
        self.conv3 = nn.Sequential(
            DepthSeperabelConv1d(
                int(256 * alpha),
                int(512 * alpha),
                3,
                stride=2,
                padding=(0, 1),
                bias=False
            ),

            DepthSeperabelConv1d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=(0, 1),
                bias=False
            ),
            DepthSeperabelConv1d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=(0, 1),
                bias=False
            ),
            DepthSeperabelConv1d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=(0, 1),
                bias=False
            ),
            DepthSeperabelConv1d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=(0, 1),
                bias=False
            ),
            DepthSeperabelConv1d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=(0, 1),
                bias=False
            )
        )

        # downsample
        self.conv4 = nn.Sequential(
            DepthSeperabelConv1d(
                int(512 * alpha),
                int(1024 * alpha),
                3,
                stride=2,
                padding=(0, 1),
                bias=False
            ),
            DepthSeperabelConv1d(
                int(1024 * alpha),
                int(1024 * alpha),
                3,
                padding=(0, 1),
                bias=False
            )
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone = backbone
        if params.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(1024, num_classes, s=16, m=0.2)
        elif params.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(1024, num_classes, s=30, m=0.5,
                                              easy_margin=params.easy_margin)
        elif params.metric == 'sphere':
            self.metric_fc = SphereProduct(1024, num_classes, m=4)
        else:
            self.metric_fc = nn.Linear(1024, num_classes)

    def forward(self, x, labels=None, train_mode=False):
        x = self.encoder(x)
        if train_mode:
            if params.metric in ['add_margin', 'arc_margin', 'sphere']:
                x = self.metric_fc(x, labels.to(torch.int64))
            else:
                x = self.metric_fc(x)

            return x
        else:
            if params.metric in ['add_margin', 'arc_margin', 'sphere']:
                logits = F.linear(F.normalize(x), F.normalize(self.metric_fc.weight))
            else:
                logits = self.metric_fc(x)
            return logits, x

    def encoder(self, x):
        x = x.unsqueeze(dim=1)
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = torch.flatten(x, 1)
        return x

    # def forward(self, x, return_feature=False):
    #     # x = x.unsqueeze(dim=1)
    #     x = self.stem(x)
    #
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = self.conv3(x)
    #     x = self.conv4(x)
    #
    #     x = self.avg(x)
    #     x = x.view(x.size(0), -1)
    #     if self.backbone:
    #         return x
    #     else:
    #         if return_feature:
    #             return self.fc(x), x  # Return the prediction and the 512-dimensional feature vector
    #         return self.fc(x)
    # x = self.fc(x)
    # return x


def mobilenet(alpha=1, class_num=10):
    return MobileNet(alpha, class_num)


if __name__ == '__main__':
    inputs = torch.rand(10, 2, 5000)
    model = MobileNet(num_classes=16)
    # model = ResNet8(num_classes=10)
    print(model)
    outputs = model(inputs)
    summary(model, (16, 2, 4800))
    # print(outputs.shape)
# data = torch.randn(10, 2, 4800)
# model = mobilenet()
# out = model(data)
# print(out.shape)
