"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/4/5 11:27 AM
"""
from re import template
import cv2
from matplotlib import image
import numpy as np
from tqdm import tqdm
import os.path as osp
import os
def template_match(ori_img_path,template_path):
    template = cv2.imread(template_path)
    ori_img = cv2.imread(ori_img_path)
    h, w = template.shape[:2]
    template = cv2.resize(template, (int(template.shape[0] / 10), int(template.shape[1] / 10)), cv2.INTER_NEAREST)
    res = cv2.matchTemplate(ori_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.63
    # 取匹配程度大于%90的坐标
    loc = np.where(res >= threshold)
    # np.where返回的坐标值(x,y)是(h,w)，注意h,w的顺序
    index = 0

    for pt in zip(*loc[::-1]):
        index = index + 1
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(ori_img, pt, bottom_right, (0, 255, 0), 5)
    end_img_path = osp.join('/Users/huhao/Downloads/result',osp.basename(ori_img_path))
    cv2.imwrite(end_img_path, ori_img)


if __name__ == '__main__':
    template_path = '/Users/huhao/Downloads/gray_template/0261P4L64392A0_L2_template.jpg'
    ori_img_path = '/Users/huhao/Downloads/resize_small/KKK0261P4L64392A0_L2-20211122_10954B.png'

    template_match(ori_img_path, template_path)
