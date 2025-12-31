import torch
import torch.nn as nn
import torch.nn.functional as F
import util.params as params
from models.metrics import *  # Assuming metric classes like AddMarginProduct, ArcMarginProduct, etc.
from torchinfo import summary


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,L] -> [N,g,C/g,L] -> [N,C/g,g,L] -> [N,C,L]'''
        N, C, L = x.size()
        g = self.groups
        return x.view(N, g, C // g, L).permute(0, 2, 1, 3).reshape(N, C, L)


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :], x[:, c:, :]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        # right
        self.conv3 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(mid_channels)
        self.conv4 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels,
                               bias=False)
        self.bn4 = nn.BatchNorm1d(mid_channels)
        self.conv5 = nn.Conv1d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm1d(mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        # right
        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        # concat
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, net_size, num_classes=16):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']

        # Input layer is now 1x24, kernel_size (2, 7) -> (1, 7)
        self.conv1 = nn.Conv1d(1, 24, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = nn.Conv1d(out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels[3])
        self.avg = nn.AdaptiveAvgPool1d(1)  # Use 1D adaptive average pool
        if params.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(out_channels[3], num_classes, s=16, m=0.2)
        elif params.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(out_channels[3], num_classes, s=30, m=0.5, easy_margin=params.easy_margin)
        elif params.metric == 'sphere':
            self.metric_fc = SphereProduct(out_channels[3], num_classes, m=4)
        else:
            self.metric_fc = nn.Linear(out_channels[3], num_classes)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
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
        out = F.relu(self.bn1(self.conv1(x)))  # Input x should have shape (batch_size, 1, seq_length)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.avg(out)  # Adaptive average pool
        out = out.view(out.size(0), -1)  # Flatten the output
        return out


# Configuration dictionary for different model sizes
configs = {
    0.5: {
        'out_channels': (48, 96, 192, 1024),
        'num_blocks': (3, 7, 3)
    },

    1: {
        'out_channels': (116, 232, 464, 1024),
        'num_blocks': (3, 7, 3)
    },
    1.5: {
        'out_channels': (176, 352, 704, 1024),
        'num_blocks': (3, 7, 3)
    },
    2: {
        'out_channels': (224, 488, 976, 2048),
        'num_blocks': (3, 7, 3)
    }
}


def shufflenet(net_size=0.5, num_classes=16):
    return ShuffleNetV2(net_size=1, num_classes=16)


if __name__ == '__main__':
    net = shufflenet()
    x = torch.randn(10, 2, 4800)
    summary(net, (16, 2, 4800))
