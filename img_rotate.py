import cv2
import numpy as np
import math

def rotate_img(img, polys, rect):
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    x1, y1 = polys[3]
    x2, y2 = polys[2]
    x3, y3 = polys[0]
    dy = y1 - y2
    dx = x1 - x2
    dhx = x1 - x3
    dhy = y1 - y3
    angle = math.degrees(math.atan2(dy, dx))
    imgW = int(math.sqrt((dx**2)+(dy**2)))
    imgH = int(math.sqrt((dhx**2)+(dhy**2)))

    if dx < 0:
        angle += 180

    height, width, channels = img.shape
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, (imgW,imgH), center)

    return img_crop

def rotate_90(img):
    rimg = img.copy()
    height, width, channels = img.shape
    chk = False
    if height > (width*1.2):
        rimg = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        chk = True

    return rimg, chk