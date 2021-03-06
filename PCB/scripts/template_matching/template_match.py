"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/17 3:43 PM
"""
import cv2
import numpy as np

def template_match(ori_img_path):
    ori_img = cv2.imread(ori_img_path)
    img_gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    template = img_gray[162:2851,227:1828]
    h, w = template.shape[:2]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    # 取匹配程度大于%80的坐标

    loc = np.where(res >= threshold)

    # np.where返回的坐标值(x,y)是(h,w)，注意h,w的顺序
    index = 0
    for pt in zip(*loc[::-1]):
        index = index +1

        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(ori_img, pt, bottom_right, (0, 255, 0), 10)
        # if index ==3:
        #     bottom_right = (pt[0] + w, pt[1] + h)
        #     cv2.rectangle(ori_img, pt, bottom_right, (0, 255, 0), 10)

    #cv2.imwrite("img.jpg", img_rgb)
    cv2.imshow('img', ori_img)
    cv2.waitKey(0)
# 1,14

def show_image(ori_img_path):
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    img = Image.open(ori_img_path)
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    ori_img_path = '/Users/huhao/Downloads/template/png/test3.png'
    template_match(ori_img_path)
    #show_image(ori_img_path)

