import torch
import torch.nn as nn
from torchinfo import summary
import torch
import torch.nn.functional as F
import util.params as params
from models.metrics import *
from models.groupconv import GroupConvShuffle
from thop import profile


class Res2NetGCModule(nn.Module):
    expansion = 1  # BasicBlock will not expand the number of channels

    def __init__(self, inplanes, planes, stride=1, downsample=None, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
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
            self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)  # Pooling for 1D data
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(GroupConvShuffle(width, width, kernel_size=3, stride=1, padding=1, groups=2))
            bns.append(nn.BatchNorm1d(width))  # BatchNorm1d instead of BatchNorm2d
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        Basic Module, there are two forms in total, 1.s=1 when the input and output dimensions are the same 2.s=2 when the size of the feature map is halved, and the dimension is doubled
        :param in_channel: Input channel number dimension
        :param stride: stride length
        :param downsample: Downsampling operation (optional)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
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
    def __init__(self, block, layers, num_classes=16, type=1, scale=4, scale_flag=False):
        super(ResNet10, self).__init__()
        self.type = type
        self.scale_flag = scale_flag
        self.scale = scale
        self.in_channels = 64

        # Change 2D convolution to 1D convolution
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)  # Change 2D BatchNorm to 1D BatchNorm
        self.relu = nn.ReLU(inplace=True)

        # Change MaxPool2d to MaxPool1d
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Make layers using the modified block and layers
        self.layer1 = self._make_layer(block, 128, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)

        # Use adaptive pooling for 1D outputs
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 256
        if params.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(self.out_dim * block.expansion, num_classes, s=params.margin_s,
                                              m=params.margin_m)
        elif params.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(self.out_dim * block.expansion, num_classes, s=30, m=0.3,
                                              easy_margin=params.easy_margin)
        elif params.metric == 'sphere':
            self.metric_fc = SphereProduct(self.out_dim * block.expansion, num_classes, m=2)
        else:
            self.metric_fc = nn.Linear(self.out_dim * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        if self.scale_flag:
            layers.append(block(self.in_channels, out_channels, stride, downsample, scale=self.scale))
        else:
            layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            if self.scale_flag:
                layers.append(block(self.in_channels, out_channels, scale=self.scale))
            else:
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=90, type=1, scale=4, scale_flag=False):
        super(ResNet18, self).__init__()
        self.type = type
        self.scale_flag = scale_flag
        self.scale = scale
        self.in_channels = 64

        # Change 2D convolution to 1D convolution
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)  # Change 2D BatchNorm to 1D BatchNorm
        self.relu = nn.ReLU(inplace=True)

        # Change MaxPool2d to MaxPool1d
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Make layers using the modified block and layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Use adaptive pooling for 1D outputs
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if params.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(512 * block.expansion, num_classes, s=16, m=0.35)
        elif params.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(512 * block.expansion, num_classes, s=30, m=0.3,
                                              easy_margin=params.easy_margin)
        elif params.metric == 'sphere':
            self.metric_fc = SphereProduct(512 * block.expansion, num_classes, m=2)
        else:
            self.metric_fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        if self.scale_flag:
            layers.append(block(self.in_channels, out_channels, stride, downsample, scale=self.scale))
        else:
            layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            if self.scale_flag:
                layers.append(block(self.in_channels, out_channels, scale=self.scale))
            else:
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
        x = self.conv1(x)
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


def res2netgc_resnet18(num_classes=16, scale=4, scale_flag=False):
    return ResNet18(Res2NetGCModule, [2, 2, 2, 2], num_classes=num_classes, type=0, scale=scale, scale_flag=scale_flag)


def resnet18(num_classes=16):
    return ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def res2netgc_resnet10(num_classes=16, scale=4, scale_flag=False):
    return ResNet10(Res2NetGCModule, [2, 2], num_classes=num_classes, type=0, scale=scale, scale_flag=scale_flag)


def resnet10(num_classes=16):
    return ResNet10(BasicBlock, layers=[1, 1], num_classes=num_classes)


if __name__ == '__main__':
    inputs = torch.rand(16, 2, 4800)
    model = resnet18(num_classes=16)
    # model = res2netgc_resnet18(scale_flag=True)
    # model = resnet10(16)
    # print(model)
    # outputs = model(inputs)
    # summary(model, (16, 2, 4800))
    flops, model_params = profile(model, (inputs,))
    summary(model, (16, 2, 4800))
    print('flops: ', flops, 'params: ', model_params)
    # print(outputs.shape)
