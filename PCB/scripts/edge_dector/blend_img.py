"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/10 9:25 PM
"""
from PIL import Image
import cv2
import os.path as osp
def fill_color_demo(img_path, end_path):

    img1 = Image.open(img_path)
    img2 = Image.open(end_path)
    image = Image.blend(img1, img2, 0.3)
    blend_end_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/scripts/graphcut/' + '20211022_10583-0-01-7_fake_B_blend.png'
    image.save(blend_end_path)
if __name__ == '__main__':
    ori_img_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/pcb_small/20211022_10583-0-01-7_fake_B.png'
    end_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/scripts/test.jpg'
    fill_color_demo(ori_img_path, end_path)