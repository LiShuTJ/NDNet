# -*- coding: utf-8 -*-
"""
Implementation code for paper:
NDNet: Space-wise Multiscale Representation Learning via Neighbor
      Decoupling for Real-time Driving Scene Parsing
      
Authors: Shu Li, Qingqing Yan, Xun Zhou, Deming Wang, Chengju Liu and Qijun Chen
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torchsummaryX import summary
import time
import os
import numpy as np
from einops import rearrange, reduce, repeat, parse_shape

from .utils import conv_bn_relu


BN_MOMENTUM = 0.1


class NeighborDecouple(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        x = rearrange(x, 'b c (h h2) (w w2) -> b (h2 w2 c) h w', h2=self.bs, w2=self.bs)
        return x
    

class NeighborCouple(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        x = rearrange(x, 'b (h2 w2 c) h w -> b c (h h2) (w w2)', h2=self.bs, w2=self.bs)
        return x
    
    
class SMFE(nn.Module):
    def __init__(self, in_ch, out_ch=None, stride=1, kernels=[1, 3, (5,1), (1,5)]):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.num_k = len(kernels)
        inter = out_ch
        outer = out_ch if stride==1 else out_ch//self.num_k
        
        self.spacewiseLearning = nn.ModuleList()
        for i in range(self.num_k):
            self.spacewiseLearning.append(conv_bn_relu(in_ch, inter, kernels[i], relu=False))
        
        self.decouple = NeighborDecouple(2)
        self.fuse = conv_bn_relu(self.num_k*inter, self.num_k*outer, 1, relu=False)

        self.couple = NeighborCouple(2) if stride==1 else nn.Identity()  
        self.downsample = None if (stride==1 and in_ch==out_ch) else conv_bn_relu(in_ch, out_ch, 1, stride=stride, relu=False)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        x  = self.decouple(x)
        g = x.chunk(self.num_k, 1)
        x = []
        for i in range(self.num_k):
            x.append(self.spacewiseLearning[i](g[i]))
        x = torch.cat(x, 1)
        x = self.fuse(F.relu(x, True))
        x = self.couple(x)
        x = F.relu(x+residual, True)
        return x


class LCGB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        inner_ch = out_ch//2
        self.down = NeighborDecouple(4)
        self.conv = nn.Sequential(
            conv_bn_relu(16*in_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
            conv_bn_relu(out_ch, out_ch, 3, groups=16),
            conv_bn_relu(out_ch, out_ch, 1),
            )

        self.avgPool = nn.AvgPool2d(9, stride=8, padding=4, count_include_pad=False)
        self.to_qkv = conv_bn_relu(out_ch, 3*inner_ch, 1)
        
        self.expand = conv_bn_relu(out_ch, out_ch, 1)
        
        
    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)

        _, _, h, w = x.shape
        pooled = self.avgPool(x)
        
        qkv = self.to_qkv(pooled).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b c (h w)'), qkv)
        out = torch.einsum('bkl,bkt->blt', [k, q])
        out = F.softmax(out*(q.shape[1]** (-0.5)), dim=1)
        out = torch.einsum('blt,btv->blv', [v, out])
        out = torch.cat([out, q], 1)
        out = rearrange(out, 'b c (h w) -> b c h w',
                                          **parse_shape(pooled, 'b c h w'))
        out = self.expand(out)

        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        x = x+out
        
        return x


class NDNet_DF2(nn.Module):
    def __init__(self, num_classes=1000, f=[64, 128, 256, 512], n=[2, 2, 2, 2]):
        super().__init__()
        
        self.num_classes = num_classes
        self.s = LCGB(3, f[0])
        self.conv3x = nn.Sequential(
            SMFE(64, 64, stride=2),
            SMFE(64, 64),
            SMFE(64, 128)
        )
        self.conv4x = nn.Sequential(
            SMFE(128, 128, stride=2),
            SMFE(128, 128),
            SMFE(128, 128),
            SMFE(128, 128),
            SMFE(128, 128),
            SMFE(128, 128),
            SMFE(128, 128),
            SMFE(128, 128),
            SMFE(128, 128),
            SMFE(128, 128),
            SMFE(128, 256)
        )
        self.conv5x = nn.Sequential(
            SMFE(256, 256, stride=2),
            SMFE(256, 256),
            SMFE(256, 256),
            SMFE(256, 256),
            SMFE(256, 512),
            SMFE(512, 512)
        )

        
        self.s8 = nn.Sequential(
                        conv_bn_relu(128, 128, 1),
                        )
        self.s16 = nn.Sequential(
                        conv_bn_relu(256, 256, 1),
                        NeighborCouple(2),
                        )
        
        self.up32 = nn.Sequential(
                        conv_bn_relu(512, 512, 1),
                        NeighborCouple(2),
                        conv_bn_relu(128, 128, 1),
                        NeighborCouple(2),
                        )
        
        self.loc = nn.Sequential(
                        conv_bn_relu(224, 128, 3)
                        )
        self.sem = conv_bn_relu(512, 128, 3)
        self.attnLoc = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        conv_bn_relu(128, 64, 1),
                        conv_bn_relu(64, 128, 1, relu=False),
                        nn.Sigmoid()
                        )
        
        self.score = nn.Conv2d(128, num_classes, 3, 1, padding=1, bias=True)
        
    
    def _make_stage(self, in_channels, out_channels, n_blocks, stride=1, block=SMFE):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name, block(in_channels, out_channels,
                                      stride=stride))
            else:
                stage.add_module(block_name,
                                 block(out_channels, out_channels, stride=1))
        return stage


    def forward(self, x):
        x = self.s(x)
        x = self.conv3x(x)
        s8 = self.s8(x)
        x = self.conv4x(x)
        s16 = self.s16(x)
        x = self.conv5x(x)
        s32 = self.up32(x)
        loc = self.loc(torch.cat([s8, s16, s32], 1))
        sem = self.sem(x)
        attn = self.attnLoc(loc)
        sem = sem*attn
        sem = F.interpolate(sem, scale_factor=4, mode='bilinear', align_corners=False)
        score = self.score(sem+loc)

        return score
    
    def init_weights(self, path):
        
        if os.path.isfile(path):
            pretrained_dict = torch.load(path)['state_dict']
            print("[INFO] LOADING PRETRAINED MODEL: ", path)
            pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                 print('=> loading {} pretrained model {}'.format(k, path))

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
    
    def convert2Inference(self):
        for n, m in self.named_modules():
            if isinstance(m, conv_bn_relu):
                m.convert2Inference()

