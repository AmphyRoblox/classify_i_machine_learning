import torch
import torch.nn as nn
import util.params as params
from models.metrics import *
from torchinfo import summary


class BasicConv1(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size=3, padding=1):
        super(BasicConv1, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class OriginNet(nn.Module):
    def __init__(self, block, layers, num_classes=90):
        super(OriginNet, self).__init__()
        self.in_channels = 64
        # Define the convolutional layer
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if params.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(1024, num_classes, s=30, m=0.35)
        elif params.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(1024, num_classes, s=30, m=0.5,
                                              easy_margin=params.easy_margin)
        elif params.metric == 'sphere':
            self.metric_fc = SphereProduct(1024, num_classes, m=2)
        else:
            self.metric_fc = nn.Linear(1024, num_classes)
        # nitialization parameters -> Affect accuracy 7%
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):

        layers = []
        layers.append(block(self.in_channels, out_channels))

        self.in_channels = out_channels
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def get_origin_model(num_classes=90):
    return OriginNet(BasicConv1, [2, 2, 2, 2], num_classes=num_classes)


if __name__ == '__main__':
    inputs = torch.rand(1, 2, 4800).cuda()
    model = get_origin_model().cuda()
    summary(model, (16, 2, 4800))
