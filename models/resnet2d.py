"""
Author: yida
Time is: 2022/1/7 19:49
this Code: Re-implement the real ResNet18, the same model as the official implementation by torch.
Reasons for not repeating:
The parameters were not initialized according to the specified method.
Specifying the initialization of the BN layer can also improve the accuracy by 1-2%.
Result: Now it can achieve the same accuracy as the official model.
A very worthwhile blog to refer to https://blog.csdn.net/weixin_44331304/article/details/106127552?spm=1001.2014.3001.5501
"""
import os

import torch
import torch.nn as nn
from torchinfo import summary

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BasicBlock_ghost(nn.Module):
    def __init__(self, in_channel, s):
        """
        Basic module, there are two forms: 1. when s=1, the input and output dimensions are the same; 2. When s=2, the size of the feature map is reduced by half, and the dimension is doubled
        :param in_channel: Dimension of the number of input channels
        :param s: s=1 means no reduction; s=2 means scale reduction
        """
        super(BasicBlock_ghost, self).__init__()
        self.s = s
        self.conv1 = nn.Conv2d(in_channel, in_channel * s, kernel_size=(1, 3), stride=s, padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel * s)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel * s, in_channel * s, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel * s)
        if self.s == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * s, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(in_channel * s)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.s == 2:  # reduce
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channel, s):
        """
        Basic module, there are two forms: 1. when s=1, the input and output dimensions are the same; 2. When s=2, the size of the feature map is reduced by half, and the dimension is doubled
        :param in_channel: Dimension of the number of input channels
        :param s: s=1 means no reduction; s=2 means scale reduction
        """
        super(BasicBlock, self).__init__()
        self.s = s
        self.conv1 = nn.Conv2d(in_channel, in_channel * s, kernel_size=(1, 3), stride=s, padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel * s)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel * s, in_channel * s, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel * s)
        if self.s == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * s, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(in_channel * s)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.s == 2:  # reduce
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet8(nn.Module):
    def __init__(self, num_classes, zero_init_residual=True):
        super(ResNet8, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=2, padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=(0, 1))
        self.layer1 = nn.Sequential(
            BasicBlock(in_channel=64, s=2),
            BasicBlock(in_channel=128, s=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channel=128, s=2),
            BasicBlock(in_channel=256, s=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Initialization parameters -> affect accuracy by 7%
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Initialize BasicBlock -> affects accuracy by 1-2%
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x.unsqueeze(dim=1))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch.nn as nn


# class Encoder(nn.Module):
#     def __init__(self, zero_init_residual=True):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=2, padding=(0, 3), bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=(0, 1))
#         self.layer1 = nn.Sequential(
#             BasicBlock(in_channel=64, s=1),
#             BasicBlock(in_channel=64, s=1),
#         )
#         self.layer2 = nn.Sequential(
#             BasicBlock(in_channel=64, s=2),
#             BasicBlock(in_channel=128, s=1),
#         )
#         self.layer3 = nn.Sequential(
#             BasicBlock(in_channel=128, s=2),
#             BasicBlock(in_channel=256, s=1),
#         )
#         self.layer4 = nn.Sequential(
#             BasicBlock(in_channel=256, s=2),
#             BasicBlock(in_channel=512, s=1),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#         # Parameter initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def forward(self, x):
#         x = self.conv1(x.unsqueeze(dim=1))
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         return x
#
#
# class ResNet18(nn.Module):
#     def __init__(self, n_class):
#         super(ResNet18, self).__init__()
#         self.encoder = Encoder()
#         self.fc = nn.Linear(512, n_class)
#
#     def forward(self, x, return_feature=False):
#         features = self.encoder(x)
#         output = self.fc(features)
#         if return_feature:
#             return output, features
#         return output
class Encoder(nn.Module):
    def __init__(self, zero_init_residual=True):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=2, padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=(0, 1))
        self.layer1 = nn.Sequential(
            BasicBlock(in_channel=64, s=1),
            BasicBlock(in_channel=64, s=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channel=64, s=2),
            BasicBlock(in_channel=128, s=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channel=128, s=2),
            BasicBlock(in_channel=256, s=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channel=256, s=2),
            BasicBlock(in_channel=512, s=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Initialization parameters -> affect accuracy by 7%
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Initialize BasicBlock -> affects accuracy by 1-2%
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, return_feature=False):
        x = self.conv1(x.unsqueeze(dim=1))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Encoder25(nn.Module):
    def __init__(self, zero_init_residual=True):
        super(Encoder25, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=2, padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=(0, 1))
        self.layer1 = nn.Sequential(
            BasicBlock(in_channel=64, s=1),
            BasicBlock(in_channel=64, s=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channel=64, s=2),
            BasicBlock(in_channel=128, s=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channel=128, s=2),
            BasicBlock(in_channel=256, s=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channel=256, s=2),
            BasicBlock(in_channel=512, s=1),
        )
        self.layer5 = nn.Sequential(
            BasicBlock(in_channel=512, s=1),
            BasicBlock(in_channel=512, s=1),
        )
        self.layer6 = nn.Sequential(
            BasicBlock(in_channel=512, s=2),
            BasicBlock(in_channel=1024, s=1),
        )
        self.layer7 = nn.Sequential(
            BasicBlock(in_channel=1024, s=1),
            BasicBlock(in_channel=1024, s=1),
        )
        self.layer8 = nn.Sequential(
            BasicBlock(in_channel=1024, s=2),
            BasicBlock(in_channel=2048, s=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Initialization parameters -> affect accuracy by 7%
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Initialize BasicBlock -> affects accuracy by 1-2%
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, return_feature=False):
        x = self.conv1(x.unsqueeze(dim=1))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResNet18(nn.Module):
    def __init__(self, n_class, zero_init_residual=True, backbone=False):
        super(ResNet18, self).__init__()
        self.encoder = Encoder()
        self.backbone = backbone
        if not backbone:
            self.fc = nn.Linear(512, n_class)

    def forward(self, x, return_feature=False):
        x = self.encoder(x)
        if self.backbone:
            return x
        else:
            if return_feature:
                return self.fc(x), x  # Return the prediction and the 512-dimensional feature vector
            return self.fc(x)


class ResNet25(nn.Module):
    def __init__(self, n_class, zero_init_residual=True):
        super(ResNet25, self).__init__()
        self.encoder = Encoder25()
        self.fc = nn.Linear(2048, n_class)

    def forward(self, x, return_feature=False):
        x = self.encoder(x)
        if return_feature:
            return self.fc(x), x  # Return the prediction and the 512-dimensional feature vector
        return self.fc(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, s=4, d=3):
        super(Bottleneck, self).__init__()
        # self.conv1 = GhostModule(inplanes, planes, kernel_size=1, dw_size=d, ratio=s, bias=False)
        # self.conv2 = GhostModule(planes, planes, kernel_size=3, dw_size=d, ratio=s,
        #                          stride=stride, padding=(0, 1), bias=False)
        # self.conv3 = GhostModule(planes, planes * 4, kernel_size=1, dw_size=d, ratio=s, bias=False)
        self.conv1 = GhostModule(inplanes, planes, kernel_size=1, dw_size=d, ratio=s)
        self.conv2 = GhostModule(planes, planes, kernel_size=3, dw_size=d, ratio=s,
                                 stride=stride)
        self.conv3 = GhostModule(planes, planes * 4, kernel_size=1, dw_size=d, ratio=s)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


if __name__ == '__main__':
    inputs = torch.rand(10, 2, 4800)
    model = ResNet18(n_class=10)
    # model = ResNet8(num_classes=10)
    print(model)
    outputs = model(inputs)
    summary(model, (16, 2, 4800))
    print(outputs.shape)
