"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
from importlib import import_module
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn as nn
import numpy as np
from torch.nn import init as init
import math
from win_util import window_partitions,window_partitionx,window_reversex
from models.batchnorm import SynchronizedBatchNorm2d
from models.base_network import BaseNetwork
from models.normalization import get_nonspade_norm_layer
import torch.nn.functional as F
import torch
import re
from PIL import Image

class conv_bench(nn.Module):
    def __init__(self, n_feat = 3, kernel_size=3, act_method=nn.ReLU, bias=False):
        super(conv_bench, self).__init__()
        self.conv1 = nn.Conv2d(6,8,1)
        self.conv2 = nn.Conv2d(8,8,1)
        self.act = act_method()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        # x2 = self.relu(x1)
        y = self.conv2(x1)
        return y

class fft_bench_complex_mlp_flops(nn.Module):
    def __init__(self, dim=6, dw=1, norm='backward', act_method=nn.ReLU, window_size=0, bias=False):
        super(fft_bench_complex_mlp_flops, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        hid_dim = dim * dw
        self.complex_weight1 = nn.Conv2d(dim*2, hid_dim*2, 1)
        self.complex_weight2 = nn.Conv2d(hid_dim*2, dim*2, 1)

        # self.complex_weight1 = nn.Conv2d(dim*2, hid_dim*2, 1, bias=bias)
        # self.complex_weight2 = nn.Conv2d(hid_dim*2, dim, 1, bias=bias)
        self.conv = nn.Conv2d(6,8,1)
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        # y = nn.Conv2d(dim*2, hid_dim*2, 1)
        y = self.complex_weight1(y)
        y = self.act_fft(y)
        y = self.complex_weight2(y)
        # self.complex_weight2 = nn.Conv2d(hid_dim*2, dim * 2, 1)

        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = self.conv(y)
        return y


class symy(nn.Module):

    def __init__(self, in_chn=3, wf=64, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4):
        super(symy, self).__init__()
        self.generator = sdmy()
    def forward(self, x):
        out = self.generator(x)
        return out

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
class ResnetBlock(nn.Module):
    def __init__(self, dim, act, kernel_size=3):
        super().__init__()
        self.act = act
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            act,
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size)
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return self.act(out)



class Convres(BaseNetwork):
    def __init__(self):
        super().__init__()
        wf = 48
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 48
        self.ndf = ndf
        norm_E = "spectralinstance"
        norm_layer = get_nonspade_norm_layer(None, norm_E)
        self.layer11 = norm_layer(nn.Conv2d(4, ndf, kw, stride=2, padding=pw))
        self.layer12 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        # 48
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        # 384
        # self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 16, kw, stride=2, padding=pw))
        self.res_0 = ResnetBlock(ndf * 8, nn.LeakyReLU(0.2, False))
        self.res_1 = ResnetBlock(ndf * 8, nn.LeakyReLU(0.2, False))
        self.res_2 = ResnetBlock(ndf * 8, nn.LeakyReLU(0.2, False))
        self.so = 4


        self.mu_make = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf * 16,ndf * 8,1)
        )

        self.mu_make_0 = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf * 16,ndf * 8,1)
        )

        # self.down = nn.AvgPool2d(2, 2)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.pad_3 = nn.ReflectionPad2d(3)
        # self.pad_1 = nn.ReflectionPad2d(1)
        self.conv_7x7 = nn.Conv2d(ndf, ndf, kernel_size=7, padding=0, bias=True)

        self.upp = nn.Conv2d(8*ndf, 16*ndf, kernel_size=1, padding=0, bias=True)

        self.conv_latent_up2 = Up_ConvBlock(8 * wf, 4 * wf)
        self.conv_latent_up3 = Up_ConvBlock(4 * wf, 2 * wf)
        self.conv_latent_up4 = Up_ConvBlock(2 * wf, 1 * wf)


    def forward(self, x, gray, white, flag):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        if flag=='low':
            gray = self.layer11(gray)  # 128
            gray = self.conv_7x7(self.pad_3(self.actvn(gray)))
            gray = self.layer2(self.actvn(gray))  # 64
            gray = self.layer3(self.actvn(gray))  # 32
            gray = self.layer4(self.actvn(gray))  # 16
            gray = self.res_0(gray)
            gray = self.res_1(gray)
            gray = self.res_2(gray)

            white = self.layer11(white)  # 128
            white = self.conv_7x7(self.pad_3(self.actvn(white)))
            white = self.layer2(self.actvn(white))  # 64
            white = self.layer3(self.actvn(white))  # 32
            white = self.layer4(self.actvn(white))  # 16
            white = self.res_0(white)
            white = self.res_1(white)
            white = self.res_2(white)

            mu = torch.cat([gray, white], dim=1)
            mu = self.mu_make_0(mu)
        else:
            x = self.layer12(x)  # 128
            x = self.conv_7x7(self.pad_3(self.actvn(x)))
            x = self.layer2(self.actvn(x))  # 64
            x = self.layer3(self.actvn(x))  # 32
            x = self.layer4(self.actvn(x))  # 16
            x = self.res_0(x)
            x = self.res_1(x)
            x = self.res_2(x)
            up = self.upp(x)
            mu = self.mu_make(up)

        latent_2 = self.conv_latent_up2(mu)  # 16
        latent_3 = self.conv_latent_up3(latent_2)  # 32
        latent_4 = self.conv_latent_up4(latent_3)  # 64
        latent_list = [latent_4, latent_3, latent_2, mu]

        return mu, latent_list

class ConvEncoderLoss(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        self.ndf = ndf
        norm_E = "spectralinstance"
        norm_layer = get_nonspade_norm_layer(None, norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        # self.layer2_1 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        # self.layer3_1 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, kw, stride=1, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        # self.layer4_1 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))
        # self.layer7 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        # self.layer8 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        self.so = s0 = 4
        self.out = norm_layer(nn.Conv2d(ndf * 8, ndf * 4, kw, stride=1, padding=0))
        self.down = nn.AvgPool2d(2,2)
        # self.global_avg = nn.AdaptiveAvgPool2d((6,6))



        self.actvn = nn.LeakyReLU(0.2, False)
        self.pad_3 = nn.ReflectionPad2d(3)
        self.pad_1 = nn.ReflectionPad2d(1)
        self.conv_7x7 = nn.Conv2d(ndf, ndf, kernel_size=7, padding=0, bias=True)
        # self.opt = opt

    def forward(self, x):

        x1 = self.layer1(x) # 128
        x2 = self.conv_7x7(self.pad_3(self.actvn(x1)))
        x3 = self.layer2(self.actvn(x2)) # 64
        # x = self.layer2_1(self.actvn(x))
        x4 = self.layer3(self.actvn(x3)) # 32
        # x = self.layer3_1(self.actvn(x))
        x5 = self.layer4(self.actvn(x4)) # 16
        # x = self.layer4_1(self.actvn(x))
        return [x1, x2, x3, x4, x5]
class EncodeMap(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.layer_final = nn.Conv2d(ndf * 8, ndf * 16, kw, stride=1, padding=pw)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        # if self.opt.crop_size >= 256:
        #     x = self.layer6(self.actvn(x))
        x = self.actvn(x)
        return self.layer_final(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar


class Up_ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.LeakyReLU(0.2, False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        '''self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            activation)'''
        # norm_layer =
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size)),
            activation,
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size)),
            activation
        )

    def forward(self, x):
        # conv1 = self.conv1(x)
        y = self.conv_block(x)
        return y


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer