
import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import _init_paths
import models
from config import config
from config import update_config

from ptflops import get_model_complexity_info

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = eval('models.'+config.MODEL.NAME)(19).cuda().eval()

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    ).cuda()
    
    macs, params = get_model_complexity_info(model, (3, 512, 1024), as_strings=False, print_per_layer_stat=False)

    
    model.convert2Inference()
    
    for _ in range(10):
        pred=model(dump_input)
    iters = 100
    torch.cuda.synchronize()
    begin = time.time()
    for _ in range(iters):
        pred=model(dump_input)
    torch.cuda.synchronize()
    end = time.time()
    k = end-begin
    print('{:<30}  {:.1f} GFlops'.format('Computational complexity: ', macs/1e9))
    print('{:<30}  {:.1f} M'.format('Number of parameters: ', params/1e6))
    print('{:<30}  {:.2f} ms'.format('Average inference time:', k/iters*1000))


if __name__ == '__main__':
    main()
