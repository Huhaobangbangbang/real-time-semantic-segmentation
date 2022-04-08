"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/4/5 8:43 PM
"""
import cv2
import numpy as np

import os
import os.path as osp
import sys
import multiprocessing as mp

import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

def resize_img():
    img = cv2.imread('/Users/huhao/Downloads/Template_img/Template_CAD/0261P4L64392A0_L2-20211122_10954B.png')
    img = cv2.resize(img, (int(img.shape[0] / 20), int(img.shape[1] / 20)), cv2.INTER_NEAREST)
    cv2.imwrite('/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/scripts/use_mask_match/get_the_bg/test.jpg',img)


def reverse_image(image):
    where_0 = np.where(image == 0)
    where_255 = np.where(image == 255)
    image[where_0] = 255
    image[where_255] = 0
    return image

def delicate_img(image):
    """ 膨胀"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    expand_pic = cv2.dilate(image, kernel)
    return expand_pic

def erode_img(image):
    """腐蚀"""
    corrosion_img = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))  ##腐蚀预处理，确定处理核的大小,矩阵操作
    image = cv2.erode(image, corrosion_img, iterations=10)  # 进行腐蚀操作
    return image


def get_box(im):
    t = 40
    epsilon = 100
    img_b = im[:, :, 0]
    img_g = im[:, :, 1]
    img_r = im[:, :, 2]
    _, mask_b = cv2.threshold(img_b, t, 255, 1)
    _, mask_g = cv2.threshold(img_g, t, 255, 1)
    _, mask_r = cv2.threshold(img_r, t, 255, 1)
    mask = cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_and(mask_b, mask_g), mask_r))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    cnt = cv2.approxPolyDP(contours[0], epsilon=epsilon, closed=True)
    cnt = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(cnt))
    return mask, box



def get_the_preprocess_img(image):

    #image_reverse = reverse_image(image)
    deli_image = delicate_img(image)
    _, box = get_box(deli_image)

    background = np.zeros((image.shape[0], image.shape[1], 3), dtype="uint8")
    re = cv2.drawContours(background, [box], 0, (255, 255, 255), -1)
    cv2.imshow("img", re)
    cv2.waitKey(0)
    return box


def fill_color_demo(img_path):
    """使用floodfill将图片最外边的白色区域涂成黑色"""
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    #mask = np.zeros([h , w ], np.uint8)
    # 为什么要加2可以这么理解：当从0行0列开始泛洪填充扫描时，mask多出来的2可以保证扫描的边界上的像素都会被处理
    mask = np.zeros([h + 2, w + 2], np.uint8)  # mask必须行和列都加2，且必须为uint8单通道阵列
    # 为什么要加2可以这么理解：当从0行0列开始泛洪填充扫描时，mask多出来的2可以保证扫描的边界上的像素都会被处理
    cv2.floodFill(image, mask, (0, 0), (0, 0, 0), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)

    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    return image




if __name__ == '__main__':
    img_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/scripts/use_mask_match/get_the_bg/test.jpg'
    #box = get_the_preprocess_img(img_path)

    image = fill_color_demo(img_path)

    box = get_the_preprocess_img(image)




