"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/4/5 8:36 PM
"""
from re import template
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
from matplotlib import image
import numpy as np
from tqdm import tqdm
import os.path as osp
import os


def template_match(ori_img_path, template_path):
    template = cv2.imread(template_path)
    ori_img = cv2.imread(ori_img_path)

    ori_img1 = ori_img[0:ori_img.shape[0], 0:16000]
    ori_img2 = ori_img[0:ori_img.shape[0], 12000:28000]
    ori_img3 = ori_img[0:ori_img.shape[0], 24000:ori_img.shape[1]]

    res1 = cv2.matchTemplate(ori_img1, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    # 取匹配程度大于%90的坐标
    loc1 = np.where(res1 >= threshold)

    res2 = cv2.matchTemplate(ori_img2, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.85
    # 取匹配程度大于%90的坐标
    loc2 = np.where(res2 >= threshold)

    res3 = cv2.matchTemplate(ori_img3, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.85
    # 取匹配程度大于%90的坐标
    loc3 = np.where(res3 >= threshold)
    ori_img = use_loc_to_rectangle(loc1, loc2, loc3, ori_img, template, ori_img1, ori_img2, ori_img3)

    print('already done')
    end_img_path = osp.join('/cloud_disk/users/huh/pcb/template_matching/result', 'KKKKKK' + osp.basename(ori_img_path))
    cv2.imwrite(end_img_path, ori_img)


def use_loc_to_rectangle(loc1, loc2, loc3, ori_img, template, ori_img1, ori_img2, ori_img3):
    h, w = template.shape[:2]
    for pt in zip(*loc1[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(ori_img1, pt, bottom_right, (0, 255, 0), 10)
    for pt in zip(*loc2[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(ori_img2, pt, bottom_right, (0, 255, 0), 10)
    for pt in zip(*loc3[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(ori_img3, pt, bottom_right, (0, 255, 0), 10)

    ori_img[0:ori_img.shape[0], 0:14000] = ori_img1[0:ori_img.shape[0], 0:14000]
    ori_img[0:ori_img.shape[0], 14000:26000] = ori_img2[0:ori_img.shape[0], 2000:14000]

    ori_img[0:ori_img.shape[0], 26000:ori_img.shape[1]] = ori_img3[0:ori_img.shape[0], 2000:ori_img3.shape[1]]

    return ori_img


if __name__ == '__main__':
    template_path = '/cloud_disk/users/huh/pcb/template_matching/Template_img/gray_template/0969P4I63661A0_L3_template.jpg'
    ori_img_path = '/cloud_disk/users/huh/pcb/template_matching/Template_img/Template_CAD/6P31075S_L5-20211119_13165A.png'

    template_match(ori_img_path, template_path)
