# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:44:56 2022

@author: LiShu
"""


import torch.nn as nn

def fuse_conv_bn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                             stride=1, padding=None, dilation=1, groups=1, deploy=False, relu=True):
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = (kernel_size[0]//2, kernel_size[1]//2)
            else:
                padding = kernel_size // 2
        if relu:
            self.nonlinear = nn.ReLU(True)
        else:
            self.nonlinear = nn.Identity()
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def convert2Inference(self):
        kernel, bias = fuse_conv_bn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                                      stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv
