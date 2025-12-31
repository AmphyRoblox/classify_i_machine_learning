import torch
import torch.nn as nn
from torchinfo import summary
import util.params as params
from models.metrics import *
from thop import profile


class DepthSeperabelConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()

        # Depthwise convolution (1D)
        self.depthwise = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size, groups=input_channels, **kwargs),
            nn.BatchNorm1d(input_channels),  # BatchNorm1d for 1D convolution
            nn.ReLU(inplace=True)
        )

        # Pointwise convolution (1D)
        self.pointwise = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 1),  # 1D pointwise conv
            nn.BatchNorm1d(output_channels),  # BatchNorm1d for 1D convolution
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BasicConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()

        # Basic 1D convolution layer
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(output_channels)  # BatchNorm1d for 1D convolution
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
            BasicConv1d(2, int(32 * alpha), 7, padding=(0, 1), bias=False),
            DepthSeperabelConv1d(
                int(32 * alpha),
                int(64 * alpha),
                3,
                padding=1,
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
                padding=1,
                bias=False
            ),
            DepthSeperabelConv1d(
                int(128 * alpha),
                int(128 * alpha),
                3,
                padding=1,
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
                padding=1,
                bias=False
            ),
            DepthSeperabelConv1d(
                int(256 * alpha),
                int(256 * alpha),
                3,
                padding=1,
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
                padding=1,
                bias=False
            ),

            DepthSeperabelConv1d(
                int(512 * alpha),
                int(512 * alpha),
                7,
                padding=1,
                bias=False
            ),
        )

        # downsample
        # self.conv4 = nn.Sequential(
        #     DepthSeperabelConv1d(
        #         int(512 * alpha),
        #         int(1024 * alpha),
        #         3,
        #         stride=2,
        #         padding=1,
        #         bias=False
        #     ),
        #     DepthSeperabelConv1d(
        #         int(1024 * alpha),
        #         int(1024 * alpha),
        #         3,
        #         padding=1,
        #         bias=False
        #     )
        # )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.backbone = backbone
        self.out_dim = int(512 * alpha)
        if params.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(self.out_dim, num_classes, s=16, m=0.2)
        elif params.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(self.out_dim, num_classes, s=30, m=0.5,
                                              easy_margin=params.easy_margin)
        elif params.metric == 'sphere':
            self.metric_fc = SphereProduct(self.out_dim, num_classes, m=4)
        else:
            self.metric_fc = nn.Linear(self.out_dim, num_classes)

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
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)

        x = self.avg(x)
        x = torch.flatten(x, 1)
        return x


def mobilenet(alpha=1, class_num=16):
    return MobileNet(alpha, class_num)


if __name__ == '__main__':
    inputs = torch.rand(10, 2, 4800)
    model = mobilenet(alpha=1, class_num=16)
    print(model)
    outputs = model(inputs)
    flops, model_params = profile(model, (inputs,))
    summary(model, (16, 2, 4800))
    print('flops: ', flops, 'params: ', model_params)
