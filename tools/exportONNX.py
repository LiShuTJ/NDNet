
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
import onnx
from onnxsim import simplify

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--weights',
                        help='pretrained weights',
                        default='',
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
    
    # load weights
    if args.weights != '':
        pretrained_dict = torch.load(args.weights)['state_dict']
        print("[INFO] LOADING PRETRAINED MODEL: ", path)
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict.keys()}
        for k, _ in pretrained_dict.items():
                print('=> loading {} pretrained model {}'.format(k, args.weights))

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # export onnx
    model.convert2Inference()
    model.eval()
    torch.onnx.export(model, dump_input, config.MODEL.NAME+".onnx",
            opset_version=12)
    
    # simplify onnx
    model = onnx.load(config.MODEL.NAME+".onnx")
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, config.MODEL.NAME+".onnx")

if __name__ == '__main__':
    main()
