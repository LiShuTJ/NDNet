
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import _init_paths
import models
import datasets
from config import config
from config import update_config



palette = np.asarray([
        (128, 64, 128),     (244, 35, 232),     (70, 70, 70),   (102, 102, 156),
        (190, 153, 153),    (153, 153, 153),    (250, 170, 30), (220, 220, 0),
        (107, 142, 35),     (152, 251, 152),    (70, 130, 180), (220, 20, 60),
        (255, 0, 0),        (0, 0, 142),        (0, 0, 70), (0, 60, 100),
        (0, 80, 100),       (0, 0, 230),        (119, 11, 32)
], dtype=np.uint8)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='Experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--img_path", 
                        type=str, 
                        default='images/1.png',
                        help='Path to image that is to be predicted')
    parser.add_argument("--save_path", 
                        type=str, 
                        default='predict.png',
                        help='Path to save the result, don\'t save if not specified.')
    parser.add_argument("--cpu", 
                        action='store_true',
                        help='Use CPU to predict, use GPU by default.')
    parser.add_argument("--noShow", 
                        action='store_true',
                        help='Use CPU to predict, use GPU by default.')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    model = eval('models.'+config.MODEL.NAME)(19)
    pretrained_dict = torch.load(config.TEST.MODEL_FILE, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    if not args.cpu:
        model = model.cuda()
    
    image = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Invalid image path!")
        return
    imageOri = image.copy()
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0)

    if not args.cpu:
        image = image.cuda()

    pred = model(image)
    pred = F.interpolate(pred, scale_factor=8, mode='bilinear', align_corners=False)
    pred = pred.argmax(1)[0].cpu().numpy()
    pred = palette[pred]
    pred = pred[:,:,::-1]

    pred = cv2.addWeighted(imageOri, 0.5, pred, 0.5, 0)
    
    if args.save_path is not None:
        cv2.imwrite(args.save_path, pred)
        print("Prediction saved at", args.save_path)

    if not args.noShow:
        cv2.imshow('pred', pred)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
