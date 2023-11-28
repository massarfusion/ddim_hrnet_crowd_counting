# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Rao Fu, RainbowSecret
# --------------------------------------------------------

import os
import pdb
import logging
import torch.nn as nn
import torch


BN_MOMENTUM = 0.1


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    """Only replce the second 3x3 Conv with the TransformerBlocker"""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, temb_channels=256):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        
        # Timestep info injection, use a FiLM-like conditioning mechanism when use_scale_shift_norm==True
        use_scale_shift_norm = True
        self.out_channels = planes * self.expansion
        self.emb_channels = temb_channels
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1, )
            ),
        )
        
        if self.out_channels == inplanes:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                inplanes, self.out_channels, 1, padding=1
            )

    def forward(self, x, temb=None):
        residual = self.skip_connection(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Injecting timestep embeddings
        emb_out = self.emb_layers(temb).type(out.dtype)
        while len(emb_out.shape) < len(out.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            out = out_norm(out) * (1 + scale) + shift
            out = out_rest(out)
        else:
            out = out + emb_out
            out = self.out_layers(out)
        # Injecting timestep embeddings

        out += residual
        out = self.relu(out)

        return out
