import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import math
import util.params as params
from models.metrics import *


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.groups
        # Ensure the channels are divisible by groups
        assert C % g == 0, "The number of channels must be divisible by groups."
        x = x.view(N, g, C // g, H, W)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(N, C, H, W)
        return x


class GroupConvShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(GroupConvShuffle, self).__init__()
        self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    groups=groups, bias=False)
        self.channel_shuffle = ChannelShuffle(groups)

    def forward(self, x):
        x = self.group_conv(x)
        x = self.channel_shuffle(x)
        return x


class Res2NetGCModule(nn.Module):
    expansion = 1  # BasicBlock will not expand the number of channels

    def __init__(self, inplanes, planes, stride=1, downsample=None, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Res2NetGCModule, self).__init__()
        width = int(inplanes / scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1))
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(GroupConvShuffle(width, width, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=2))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        out = x

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2NetModule(nn.Module):
    expansion = 1  # BasicBlock will not expand the number of channels

    def __init__(self, inplanes, planes, stride=1, downsample=None, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Res2NetModule, self).__init__()
        width = int(inplanes / scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1))
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        out = x

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise Convolution: Each input channel undergoes a separate convolution operation
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, kernel_size),
            stride=stride, padding=(0, padding), groups=in_channels, bias=bias
        )

        # Pointwise Convolution: Combine the features of each channel using 1x1 convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)  # Depthwise convolution
        x = self.pointwise(x)  # Pointwise convolution
        return x


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


class GhostDSModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostDSModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # self.primary_conv = nn.Sequential(
        #     nn.Conv2d(inp, init_channels, (1, kernel_size), stride, (0, kernel_size // 2), bias=False),
        #     nn.BatchNorm2d(init_channels),
        #     nn.ReLU(inplace=True) if relu else nn.Sequential(),
        # )
        self.primary_conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(inp, inp, (1, kernel_size), stride, (0, kernel_size // 2), groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),

            # Pointwise convolution
            nn.Conv2d(inp, init_channels, 1, 1, 0, bias=False),
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


class BasicBlockGhost(nn.Module):
    expansion = 1  # BasicBlock will not expand the number of channels

    def __init__(self, inplanes, planes, stride=1, downsample=None, s=4, d=3):
        super(BasicBlockGhost, self).__init__()
        self.conv1 = GhostModule(inplanes, planes, kernel_size=3, dw_size=d, ratio=s,
                                 stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = GhostModule(planes, planes, kernel_size=3, dw_size=d, ratio=s,
                                 stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # Reserve the input for residual connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # If downsampling is needed, downsample the residual

        out += residual  # Residual connection
        out = self.relu(out)

        return out


class BasicBlockGhostDS(nn.Module):
    expansion = 1  # BasicBlock will not expand the number of channels

    def __init__(self, inplanes, planes, stride=1, downsample=None, s=4, d=3):
        super(BasicBlockGhostDS, self).__init__()
        self.conv1 = GhostDSModule(inplanes, planes, kernel_size=3, dw_size=d, ratio=s,
                                   stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = GhostDSModule(planes, planes, kernel_size=3, dw_size=d, ratio=s,
                                   stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # Reserve the input for residual connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # If downsampling is needed, downsample the residual

        out += residual  # Residual connection
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        Basic module, there are two forms in total, 1.s=1 when the input and output dimensions are the same 2.s=2 when the size of the feature map is halved, and the dimension is doubled.
        :param in_channel: Input channel count dimension
        :param s: s=1 do not reduce the scale s=2 reduce the scale
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 3), stride=stride, padding=(0, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
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


class ResNet10(nn.Module):
    def __init__(self, block, layers, num_classes=90):
        super(ResNet10, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if params.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(512 * block.expansion, num_classes, s=16, m=0.2)
        elif params.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(512 * block.expansion, num_classes, s=30, m=0.5,
                                              easy_margin=params.easy_margin)
        elif params.metric == 'sphere':
            self.metric_fc = SphereProduct(512 * block.expansion, num_classes, m=4)
        else:
            self.metric_fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialization parameters -> affecting accuracy by 7%
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
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
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
        x = self.conv1(x.unsqueeze(dim=1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=90):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=2, padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=2, padding=(0, 1))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[1], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if params.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(512 * block.expansion, num_classes, s=30, m=0.35)
        elif params.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(512 * block.expansion, num_classes, s=30, m=0.5,
                                              easy_margin=params.easy_margin)
        elif params.metric == 'sphere':
            self.metric_fc = SphereProduct(512 * block.expansion, num_classes, m=4)
        else:
            self.metric_fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialization parameters -> affecting accuracy by 7%
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
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
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
        x = self.conv1(x.unsqueeze(dim=1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def resnet18(num_classes=90):
    return ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet10(num_classes=16):
    return ResNet10(BasicBlock, [2, 2, 2], num_classes=num_classes)


def ghost_resnet10(num_classes=16):
    return ResNet10(BasicBlockGhost, [2, 2, 2], num_classes=num_classes)


def ghostds_resnet10(num_classes=16):
    return ResNet10(BasicBlockGhostDS, [2, 2, 2], num_classes=num_classes)


def ghost_resnet18(num_classes=90):
    return ResNet18(BasicBlockGhost, [2, 2, 2, 2], num_classes=num_classes)


def ghostds_resnet18(num_classes=90):
    return ResNet18(BasicBlockGhostDS, [2, 2, 2, 2], num_classes=num_classes)


def res2net_resnet10(num_classes=16):
    return ResNet10(Res2NetModule, [2, 2, 2], num_classes=num_classes)


def res2netgc_resnet10(num_classes=16):
    return ResNet10(Res2NetGCModule, [2, 2, 2], num_classes=num_classes)


if __name__ == '__main__':
    inputs = torch.rand(1, 2, 4800).cuda()
    labels = torch.randint(0, 16, (1, 1)).cuda()
    model = resnet18(num_classes=16).cuda()
    # model1 = ghost_resnet18(num_classes=16).cuda()
    # model2 = ghostds_resnet18(num_classes=16).cuda()
    model1 = ghost_resnet10(num_classes=16).cuda()
    model2 = ghostds_resnet10(num_classes=16).cuda()
    model3 = res2net_resnet10(num_classes=16).cuda()
    model4 = res2netgc_resnet10(num_classes=16).cuda()

    # model = ResNet8(num_classes=10)
    # print(model)
    # outputs = model(inputs, labels=labels, train_mode=True)
    summary(model, (16, 2, 4800))
    summary(model1, (16, 2, 4800))
    summary(model2, (16, 2, 4800))
    summary(model3, (16, 2, 4800))
    summary(model4, (16, 2, 4800))
    # print(outputs.shape)
