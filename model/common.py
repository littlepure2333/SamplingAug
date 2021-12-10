import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def valid_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=0, bias=bias)

def adaConv(input, kernel, bias=None):
    input = F.pad(input,(1,1,1,1)) # same padding
    B,C_in,H,W = input.shape
    B,C_out,C_in,h,w = kernel.shape
    H_out = H - h + 1
    W_out = W - w + 1

    inp_unf = torch.nn.functional.unfold(input, (h,w))
    out_unf = inp_unf.transpose(1,2) # (B, H_out*W_out, C_in*h*w)
    w_tran = kernel.view(kernel.size(0),kernel.size(1),-1).transpose(1,2) # (B, C_in*h*w, C_out)
    out_unf = out_unf.matmul(w_tran).transpose(1,2) # (B, C_out, H_out*W_out)
    out = out_unf.view(B,C_out,H_out,W_out)
    b = bias.reshape(B,C_out,1,1).repeat(1,1,H_out,W_out)
    out = out + b

    return out

# class AdaConv(nn.Module):
#     def __init__(self, kernel_size):
#         super(AdaConv, self).__init__()
#         self.weight = None
#         self.bias = None
#         self.padding = (kernel_size//2)
    
#     def update_param(self, weight, bias):
#         self.weight = weight
#         self.bias = bias
    
#     def forward(self, x):
#         out = F.conv2d(x, weight=self.weight, bias=self.bias, 
#                        stride=1, padding=self.padding)
#         return out

class Space2Batch(nn.Module):
    def __init__(self, kernel_size):
        super(Space2Batch, self).__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = self.kernel_size - (H % self.kernel_size)
        pad_w = self.kernel_size - (W % self.kernel_size)

        x = F.pad(x, (0, pad_w, 0, pad_h)) # (B,C,H,W)
        x = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        
        x = x.permute(0,2,1)
        x = x.contiguous().view(x.shape[0]*x.shape[1], C, self.kernel_size, self.kernel_size)
        # (B*N, C, h, w), N is the number of patches

        return x, B, C, H, W, pad_h, pad_w

class Batch2Space(nn.Module):
    def __init__(self, kernel_size):
        super(Batch2Space, self).__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size

    def forward(self, x, B, C, H, W):
        # (B*N, C, h, w), N is the number of patches
        x = x.contiguous().view(B, int(x.shape[0]/B), -1)
        x = x.permute(0,2,1)
        x = F.fold(x, output_size=(H,W), kernel_size=self.kernel_size, stride=self.stride)
        # (B, C, H, W)

        return x


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, out_channels, conv = default_conv,
         bias=True, bn=True, act=nn.LeakyReLU(0.2, True), res_scale=1):

        super(Bottleneck, self).__init__()
        m_res = []
        # conv1
        m_res.append(conv(in_channels, out_channels//4, 1, bias=bias))
        if bn:
            m_res.append(nn.BatchNorm2d(out_channels//4))
        m_res.append(act)
        # conv2
        m_res.append(conv(out_channels//4, out_channels//4, 3, bias=bias))
        if bn:
            m_res.append(nn.BatchNorm2d(out_channels//4))
        m_res.append(act)
        # conv3
        m_res.append(conv(out_channels//4, out_channels, 1, bias=bias))
        if bn:
            m_res.append(nn.BatchNorm2d(out_channels))

        self.res = nn.Sequential(*m_res)
        self.res_scale = res_scale

        m_shortcut = [conv(in_channels, out_channels, 1, bias=bias)]
        if bn:
            m_shortcut.append(nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Sequential(*m_shortcut)

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        res += self.shortcut(x)

        return res

class MyResBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(MyResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act:
            m.append(act)
        m.append(conv(out_channels, out_channels, kernel_size, bias=bias))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        self.identity = conv(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x):
        res = self.body(x)
        x = self.identity(x)
        x += res

        return x

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

