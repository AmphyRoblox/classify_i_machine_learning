# 2022.09.30-Changed for building Ghost-ResNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a Ghost-ResNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, (1, kernel_size), stride, (0, kernel_size // 2), bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, (1, dw_size), 1, (0, dw_size // 2), groups=init_channels,
                      bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class BasicBlock(nn.Module):
    expansion = 1  # BasicBlock will not expand the number of channels

    def __init__(self, inplanes, planes, stride=1, downsample=None, s=4, d=3):
        super(BasicBlock, self).__init__()
        # First convolutional layer: 3x3 convolution
        # self.conv1 = GhostModule(inplanes, planes, kernel_size=3, dw_size=d, ratio=s,
        #                          stride=stride, padding=1, bias=False)
        self.conv1 = GhostModule(inplanes, planes, kernel_size=3, dw_size=d, ratio=s,
                                 stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolutional layer: 3x3 convolution
        # self.conv2 = GhostModule(planes, planes, kernel_size=3, dw_size=d, ratio=s,
        #                          stride=1, padding=1, bias=False)
        self.conv2 = GhostModule(planes, planes, kernel_size=3, dw_size=d, ratio=s,
                                 stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # Reserve the input for residual connections

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # If downsampling is needed, downsample the residual.

        out += residual  # Residual connection
        out = self.relu(out)

        return out


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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=90, s=4, d=3, backbone=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=2, padding=(0, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=(0, 1))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, s=s, d=d)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, s=s, d=d)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, s=s, d=d)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, s=s, d=d)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone = backbone
        if not backbone:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not isinstance(m, GhostModule):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, s=4, d=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # GhostModule(self.inplanes, planes * block.expansion, ratio=s, dw_size=d,
                #             kernel_size=1, stride=stride, bias=False),
                GhostModule(self.inplanes, planes * block.expansion, ratio=s, dw_size=d,
                            kernel_size=1, stride=stride),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, s, d))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, s=s, d=d))

        return nn.Sequential(*layers)

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
        if self.backbone:
            return x
        else:
            if return_feature:
                return self.fc(x), x  # Return the prediction and the 512-dimensional feature vector
            return self.fc(x)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet18_ghost(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet18_ghost_backbone(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], backbone=True, **kwargs)
    return model


if __name__ == "__main__":
    # Create random input (assuming it is an RGB image with a batch size of 1 and a height and width of 224x224)
    x = torch.randn(1, 2, 4800)

    # Instantiate the ResNet50 model
    # model = resnet50(num_classes=90)  # Suppose 1000 categories are output
    model = resnet18_ghost(num_classes=90)
    model.eval()  # Switch the model to evaluation mode

    # Forward propagation to obtain the model output
    output = model(x)

    # The shape of the printed output
    print("Output shape：", output.shape)  # The expected output shape is [1, 1000]
