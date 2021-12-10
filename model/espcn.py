import torch
import torch.nn as nn
import torch.nn.init as init
import cv2
import numpy as np
from model import common


def make_model(args, parent=False):
    return ESPCN(args)


class ESPCN(nn.Module):
    """ESPCN baseline"""
    def __init__(self, args):
        super(ESPCN, self).__init__()
        # self.act_func = nn.LeakyReLU(negative_slope=0.2)
        self.act_func = nn.ReLU(inplace=True)
        self.scale = int(args.scale[0])  # use scale[0]
        self.n_colors = args.n_colors

        self.conv1 = nn.Conv2d(self.n_colors, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

        # base_c1, base_c2, expand_r = 64, 32, 4
        # self.conv1 = nn.Conv2d(n_colors, base_c1 * expand_r, (5, 5), (1, 1), (2, 2))
        # self.conv2 = nn.Conv2d(base_c1 * expand_r, base_c1 * expand_r, (3, 3), (1, 1), (1, 1))
        # self.conv3 = nn.Conv2d(base_c1 * expand_r, base_c2 * expand_r, (3, 3), (1, 1), (1, 1))
        # self.conv4 = nn.Conv2d(base_c2 * expand_r, n_colors * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        # self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x, krl=None):
        # (x, krl)
        out = self.act_func(self.conv1(x))
        out = self.act_func(self.conv2(out))
        out = self.act_func(self.conv3(out))
        out = self.pixel_shuffle(self.conv4(out))
        return out

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


if __name__ == '__main__':
    pass


################################
# class ESPCNCond(nn.Module):
#     def __init__(self, args):
#         super(ESPCNCond, self).__init__()
#         # self.act_func = nn.LeakyReLU(negative_slope=0.2)
#         self.act_func = nn.ReLU(inplace=True)
#         upscale_factor = args.scale[0]
#         n_colors = args.n_colors
#
#         n = 8
#         self.conv1 = common.CondConv2d(in_channels=n_colors, out_channels=64, kernel_size=5, stride=1, padding=2, num=n)
#         self.conv2 = common.CondConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, num=n)
#         self.conv3 = common.CondConv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, num=n)
#         self.conv4 = common.CondConv2d(in_channels=32, out_channels=3 * (upscale_factor ** 2), kernel_size=3,
#                                        stride=1, padding=1, num=n)
#
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#         self._initialize_weights()
#
#     def forward(self, x, krl=None):
#         x = self.act_func(self.conv1(x))
#         x = self.act_func(self.conv2(x))
#         x = self.act_func(self.conv3(x))
#         x = self.pixel_shuffle(self.conv4(x))
#         return x
#
#     def _initialize_weights(self):
#         self.conv1.initialize_weights(init.calculate_gain('relu'))
#         self.conv2.initialize_weights(init.calculate_gain('relu'))
#         self.conv3.initialize_weights(init.calculate_gain('relu'))
#         self.conv4.initialize_weights()


# class ESPCNMeta(nn.Module):
#     def __init__(self, args):
#         super(ESPCNMeta, self).__init__()
#         # self.act_func = nn.LeakyReLU(negative_slope=0.2)
#         self.act_func = nn.ReLU(inplace=True)
#         upscale_factor = args.scale[0]
#         n_colors = args.n_colors
#
#         n = 4
#         self.kernel_net = common.KernelNet(num=n)
#         self.conv1 = common.MetaConv2d(in_channels=n_colors, out_channels=64, kernel_size=5, stride=1, padding=2, num=n)
#         self.conv2 = common.MetaConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, num=n)
#         self.conv3 = common.MetaConv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, num=n)
#         self.conv4 = common.MetaConv2d(in_channels=32, out_channels=n_colors * (upscale_factor ** 2), kernel_size=3,
#                                        stride=1, padding=1, num=n)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#         self._initialize_weights()
#
#     def forward(self, x):
#         meta_ft = self.kernel_net(x)
#         x = self.act_func(self.conv1(x, meta_ft))
#         x = self.act_func(self.conv2(x, meta_ft))
#         x = self.act_func(self.conv3(x, meta_ft))
#         x = self.pixel_shuffle(self.conv4(x, meta_ft))
#         return x
#
#     def _initialize_weights(self):
#         self.conv1.initialize_weights(init.calculate_gain('relu'))
#         self.conv2.initialize_weights(init.calculate_gain('relu'))
#         self.conv3.initialize_weights(init.calculate_gain('relu'))
#         self.conv4.initialize_weights()


# class ESPCNKrl(nn.Module):
#     def __init__(self, args):
#         super(ESPCNKrl, self).__init__()
#         # self.act_func = nn.LeakyReLU(negative_slope=0.2)
#         self.act_func = nn.ReLU(inplace=True)
#         upscale_factor = args.scale[0]
#         n_colors = args.n_colors
#
#         n = 4
#         self.conv1 = common.KrlConv2d(in_channels=n_colors, out_channels=64, kernel_size=5, stride=1, padding=2, num=n)
#         self.conv2 = common.KrlConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, num=n)
#         self.conv3 = common.KrlConv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, num=n)
#         self.conv4 = common.KrlConv2d(in_channels=32, out_channels=n_colors * (upscale_factor ** 2), kernel_size=3,
#                                       stride=1, padding=1, num=n)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#         self._initialize_weights()
#
#     def forward(self, x, krl=None):
#         x = self.act_func(self.conv1(x, krl))  # krl is B*K
#         x = self.act_func(self.conv2(x, krl))
#         x = self.act_func(self.conv3(x, krl))
#         x = self.pixel_shuffle(self.conv4(x, krl))
#         return x
#
#     def _initialize_weights(self):
#         self.conv1.initialize_weights(init.calculate_gain('relu'))
#         self.conv2.initialize_weights(init.calculate_gain('relu'))
#         self.conv3.initialize_weights(init.calculate_gain('relu'))
#         self.conv4.initialize_weights()


# class ESPCNKSFT(nn.Module):
#     def __init__(self, args):
#         super(ESPCNKSFT, self).__init__()
#         # self.act_func = nn.LeakyReLU(negative_slope=0.2)
#         self.act_func = nn.ReLU(inplace=True)
#         upscale_factor = args.scale[0]
#         n_colors = args.n_colors
#
#         self.conv1 = nn.Conv2d(n_colors, 64, (5, 5), (1, 1), (2, 2))
#         self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
#         self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
#         self.conv4 = nn.Conv2d(32, n_colors * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
#
#         self.sft1 = common.KSFT_Layer(nf=64, para=21)
#         self.sft2 = common.KSFT_Layer(nf=64, para=21)
#         self.sft3 = common.KSFT_Layer(nf=32, para=21)
#
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#
#     def forward(self, x, krl=None):
#         krl_map = krl
#
#         x = self.act_func(self.sft1(self.conv1(x), krl_map))
#         x = self.act_func(self.sft2(self.conv2(x), krl_map))
#         x = self.act_func(self.sft3(self.conv3(x), krl_map))
#         x = self.pixel_shuffle(self.conv4(x))
#         return x

# class ESPCN(nn.Module):
#     def __init__(self, args):
#         super(ESPCN, self).__init__()
#         # self.act_func = nn.LeakyReLU(negative_slope=0.2)
#         self.act_func = nn.ReLU(inplace=True)
#         self.scale = args.scale[0]
#         upscale_factor = args.scale[0]
#         n_colors = args.n_colors
#
#         self.conv1 = common.GhostBlockPlus(n_colors,256,kernel_size=5,stride=1,ratio=4)
#         self.conv2 = common.GhostBlockPlus(256,256,kernel_size=3,stride=1,ratio=4)
#         self.conv3 = common.GhostBlockPlus(256,32,kernel_size=3,stride=1,ratio=4)
#         self.conv4 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#         self._initialize_weights()
#
#     def forward(self, x, krl=None):
#         out = self.act_func(self.conv1(x))
#         out = self.act_func(self.conv2(out))
#         out = self.act_func(self.conv3(out))
#         out = self.pixel_shuffle(self.conv4(out))
#         return out
#
#     def _initialize_weights(self):
#         init.orthogonal_(self.conv1.primary_conv.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv2.primary_conv.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv3.primary_conv.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv4.weight)

