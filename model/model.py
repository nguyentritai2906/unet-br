from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        depthwise = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size,
                              stride=stride,
                              padding=dilation,
                              dilation=dilation,
                              groups=in_channels,
                              bias=bias)
        bn_depth = norm_layer(in_channels)
        pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        bn_point = norm_layer(out_channels)

        self.block = nn.Sequential(
            OrderedDict([('depthwise', depthwise), ('bn_depth', bn_depth),
                         ('relu1', nn.ReLU(inplace=True)),
                         ('pointwise', pointwise), ('bn_point', bn_point),
                         ('relu2', nn.ReLU(inplace=True))]))

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            SeparableConv2d(in_channels, mid_channels, kernel_size=3),
            SeparableConv2d(mid_channels, out_channels, kernel_size=3))
        self.residual = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        doubleconv = self.double_conv(x)
        x = self.residual(x)
        output = torch.add(x, doubleconv)
        return output


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 bilinear=True,
                 pad_to_size=True):
        super().__init__()

        self.pad_to_size = pad_to_size
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bicubic',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if self.pad_to_size:
            x1 = pad_tensor(x2, x1)
        else:
            x2 = crop_tensor(x1, x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """ Full assembly of the parts to form the complete network """
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256 // factor)

        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)

        x = self.up1(x3, x2)
        x = self.dropout(x)
        x = self.up2(x, x1)
        x = self.dropout(x)
        logits = self.outc(x)
        return logits


class UNetBR(nn.Module):
    def __init__(self, num_block):
        super(UNetBR, self).__init__()
        self.num_block = num_block
        blocks = []
        for _ in range(num_block):
            blocks.append(UNet(1))
        self.blocks = nn.Sequential(*blocks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        outputs = []
        for block in self.blocks:
            output = block(input)
            output = torch.add(input, output)
            output = self.sigmoid(output)
            outputs.append(output)
            input = output

        return outputs


def crop_tensor(target_tensor, tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


def pad_tensor(target_tensor, tensor):
    # input is CHW
    diffY = target_tensor.size()[2] - tensor.size()[2]
    diffX = target_tensor.size()[3] - tensor.size()[3]

    tensor = F.pad(
        tensor,
        [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    return tensor
