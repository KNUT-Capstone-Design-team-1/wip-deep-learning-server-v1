"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from CRAFT_pytorch import craft_utils, imgproc, file_utils
from CRAFT_pytorch.craft import CRAFT
import json
import zipfile

import img_rotate

from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='CRAFT_pytorch/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.4, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.1, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.0, type=float, help='image magnification ratio')
parser.add_argument('--pill_folder', default='./pill_image/', type=str, help='folder path to input images')
canvas_size = 1280
mag_ratio = 0.2

def test_net(net, image, text_threshold, link_threshold, low_text):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    # forward pass
    with torch.no_grad():
        y, feature = net(x) # result Coordinates

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    return boxes, polys

def crop_img(img, boxes):
    imgCrops = []
    dst = img.copy()
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32)
        rectBox = box.astype(np.int32).reshape(4,1,2)
        rect = cv2.minAreaRect(rectBox)
        try:
            imgCrop = img_rotate.rotate_img(dst, poly, rect)
            imgCrop, _ = img_rotate.rotate_90(imgCrop)
            imgCrops.append(imgCrop)
        except:
            print('error')
    return imgCrops


def detect_text_img():
    torch.set_num_threads(2)
    args = parser.parse_args()
    # load net
    net = CRAFT()     # initialize
    
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    net.eval()

    image_list, _, _ = file_utils.get_files(args.pill_folder)
    try:
        image_path = image_list[0]

        image = imgproc.loadImage(image_path)

        bboxes, polys = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text)

        crop_files = crop_img(image[:,:,::-1], polys)

        return crop_files
    except:
        print('no image')