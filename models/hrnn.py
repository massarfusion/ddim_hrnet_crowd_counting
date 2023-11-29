import numpy as np
import os, yaml
import torch as torch
import math
import torch
import torch.nn as nn
from .hr import HighResolutionTransformer, HighResolutionNet
from .hr import UNet
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    
    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
    
    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch = config.model.ch, config.model.out_ch  # ch is in-unet channel number
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size  # tuple (H, W)
        
        
        # #--- HRT(Attn) ---
        # cwd_stack = os.getcwd()
        # os.chdir("models/hr")
        # with open('./config/hrt/hrt_small.yaml', "r") as f:
        #     yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        # ode = yaml_cfg['MODEL']['HRT']
        # model = HighResolutionTransformer(ode, num_classes=1000)
        # assert os.path.exists("./weight/hrt_small.pth")
        # ckpt = torch.load("./weight/hrt_small.pth", map_location='cpu')
        # model.load_state_dict(ckpt['model'])
        # os.chdir(cwd_stack)
        # self.backbone = model
        # #--- HRT ---
        
        # #--- HRNet(CNN) ---
        # cwd_stack = os.getcwd()
        # os.chdir("./models/hr/")
        # with open('./config/hrnet/hrnet_w32.yaml', "r") as f:
        #     yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        #
        # ode = yaml_cfg['MODEL']['HRNET']
        # model = HighResolutionNet(ode, num_classes=1000)
        # assert os.path.exists(
        #     "./weight/hrnetv2_w32_imagenet_pretrained.pth"), "Have you collected this 'hrnetv2_w32_imagenet_pretrained' file? "
        # ckpt = torch.load("./weight/hrnetv2_w32_imagenet_pretrained.pth", map_location='cpu')
        # model.load_state_dict(ckpt, strict=False)
        # os.chdir(cwd_stack)
        # self.backbone = model
        # #--- HRNet ---
        
        # #--- HRNet(UNet)  w32 ---
        cwd_stack = os.getcwd()
        os.chdir("./models/hr/")
        with open('./config/hrnet/hrnet_w32.yaml', "r") as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        ode = yaml_cfg['MODEL']['HRNET']
        model = UNet(ode, num_classes=1000)
        assert os.path.exists(
            "./weight/hrnetv2_w32_imagenet_pretrained.pth"), "Have you collected this 'hrnetv2_w32_imagenet_pretrained' file? "
        ckpt = torch.load("./weight/hrnetv2_w32_imagenet_pretrained.pth", map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        os.chdir(cwd_stack)
        self.backbone = model
        # #--- HRNet ---
        
        # timestep embedding
        self.ch = yaml_cfg['MODEL']['HRNET']['STAGE4']['NUM_CHANNELS'][0]
        self.temb_ch = yaml_cfg['MODEL']['HRNET']['STAGE4']['NUM_CHANNELS'][-1]
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])
        
        # output heads:
        self.reg_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(480, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.density_head = nn.Sequential(
            nn.Conv2d(128, 1, 1, ),
            nn.ReLU()
        )
    
    def forward(self, x, t):
        """
                Model takes x and t as input, x is noised map, t is time information
                x is concatenation of [RGB, noised_heatmap] on dim=1(channel dim), input channels in 4
        """
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        y_list = self.backbone(x, temb)
        _, _, h, w = y_list[0].size()
        feat1 = y_list[0]
        feat2 = F.interpolate(y_list[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(y_list[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(y_list[3], size=(h, w), mode="bilinear", align_corners=True)
        
        feats = torch.cat([feat1, feat2, feat3, feat4], 1)  # 480 channels
        out_reg = self.reg_head(feats)
        out_density=self.density_head(out_reg)
        return out_density
        # out_aux = F.interpolate(
        #     out_aux, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=True
        # )
        
        '''HRNet v2 w32
        INPUT = torch.randn(2,3,768,1024)
        0 feature map before classification head is shaped as torch.Size([2, 32, 192, 256])
        1 feature map before classification head is shaped as torch.Size([2, 64, 96, 128])
        2 feature map before classification head is shaped as torch.Size([2, 128, 48, 64])
        3 feature map before classification head is shaped as torch.Size([2, 256, 24, 32])
        '''
        
        ''' 输出头
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        out_aux = F.interpolate(
                out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
            )

        '''
        
        
        
        assert False, "Not implemented  out-way processor"
