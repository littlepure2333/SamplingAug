import torch
import torch.nn as nn
import torch.nn.init as init
from model import common

"""Image Super-Resolution Using Deep Convolutional Networks"""


def make_model(args, parent=False):
    return SRCNN(args)
    # return SRCNNCond(args)


class SRCNN(nn.Module):
    def __init__(self, args):
        super(SRCNN, self).__init__()
        n_colors = args.n_colors
        self.scale = args.scale[0]
        self.upsample = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=False)
        self.conv1 = nn.Conv2d(n_colors, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, n_colors, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, krl=None):
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class SRCNNCond(nn.Module):
    def __init__(self, args):
        super(SRCNNCond, self).__init__()
        n_colors = args.n_colors
        n = 4
        self.conv1 = common.CondConv2d(in_channels=n_colors, out_channels=64, kernel_size=9, stride=1, padding=9//2,
                                       num=n)
        self.conv2 = common.CondConv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=5//2, num=n)
        self.conv3 = common.CondConv2d(in_channels=32, out_channels=n_colors, kernel_size=5, stride=1, padding=5//2,
                                       num=n)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x, krl=None):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        self.conv1.initialize_weights(init.calculate_gain('relu'))
        self.conv2.initialize_weights(init.calculate_gain('relu'))
        self.conv3.initialize_weights(1)


if __name__ == '__main__':
    print(__file__)