import torch
from complexNN.nn import *
import torch.nn as nn
from models.metrics import *
import util.params as params
from torchsummaryX import summary


class CBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        Basic module, there are two forms: 1. When s=1, the input and output dimensions are the same; 2. When s=2, the size of the feature map is reduced by half, and the dimension is doubled.
        :param in_channel: Dimension of the number of input channels
        :param s: s=1 means no reduction; s=2 means reducing the scale
        """
        super(CBasicBlock, self).__init__()
        self.conv1 = cConv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = cBatchNorm1d(out_channel)
        self.relu = cRelu()
        self.conv2 = cConv1d(out_channel, out_channel, kernel_size=3, stride=1, padding='same')
        self.bn2 = cBatchNorm1d(out_channel)

        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 3), stride=stride, padding=(0, 1),
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channel)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=16):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = cConv1d(1, 64, kernel_size=7, stride=1, padding='same', bias=False)
        self.bn1 = cBatchNorm1d(64)
        self.relu = cRelu()
        self.maxpool = cMaxPool1d(3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[1], stride=2)
        self.avgpool = cAvgPool1d(300)
        self.metric_fc = cLinear(512 * block.expansion, num_classes)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # if params.metric == 'add_margin':
        #     self.metric_fc = AddMarginProduct(512 * block.expansion * 2, num_classes, s=30, m=0.35)
        # elif params.metric == 'arc_margin':
        #     self.metric_fc = ArcMarginProduct(512 * block.expansion * 2, num_classes, s=30, m=0.5,
        #                                       easy_margin=params.easy_margin)
        # elif params.metric == 'sphere':
        #     self.metric_fc = SphereProduct(512 * block.expansion * 2, num_classes, m=4)
        # else:
        #     self.metric_fc = nn.Linear(512 * block.expansion * 2, num_classes)

        # Initialization parameters -> affect accuracy by 7%
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                cConv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                cBatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None, train_mode=False):
        x = self.encoder(x)
        if train_mode:
            return self.metric_fc(x)
        else:
            logits = self.metric_fc(x)
            return logits, x

    def encoder(self, x):
        x = torch.complex(x[:, 0, :], x[:, 1, :])
        x = self.conv1(x.unsqueeze(1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # real_part = torch.unsqueeze(x.real, dim=2)
        # imag_part = torch.unsqueeze(x.imag, dim=2)
        # x_combined = torch.cat((real_part, imag_part), dim=1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def complex_resnet18(num_classes=16):
    return ResNet18(CBasicBlock, [2, 2, 2, 2], num_classes=num_classes)


if __name__ == '__main__':
    # batch_size, in_channels, out_channels, seq_len = 10, 1, 16, 4800
    # conv_tensor = torch.rand((batch_size, in_channels, seq_len))
    # conv1d = cConv1d(in_channels, out_channels, padding='same')
    # basic_1d = CBasicBlock(in_channels, out_channels)
    # print(conv1d(conv_tensor).shape)
    # print(basic_1d(conv_tensor).shape)
    inputs = torch.rand(1, 2, 4800)
    model = complex_resnet18(16)
    model(inputs)
    # summary(model, inputs)
