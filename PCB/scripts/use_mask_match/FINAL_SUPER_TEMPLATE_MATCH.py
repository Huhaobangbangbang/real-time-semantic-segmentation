"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/4/8 11:13 PM
"""
import numpy as np
import matplotlib.pyplot as plt
"""本脚本适用于超大图像纵向识别"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import os.path as osp
from toml import load

def get_template(img_path,pattern0_bbox):
    """通过pattern_bbox得到位置信息"""
    image = cv2.imread(img_path)
    template = image[pattern0_bbox[1]:pattern0_bbox[3],pattern0_bbox[0]:pattern0_bbox[2]]
    return template


def get_only_template_area(loc, blackground_mask,img_tmp,template_image,k):
    """通过分割后的原图得到 仅包含模板的mask图"""
    h, w = template_image.shape[:2]
    for index in range(k+1):
        for pt in zip(*loc[index][::-1]):
            bottom_right = (pt[0] + w, pt[1] + h)
            blackground_mask[pt[1]+12000*index:bottom_right[1]+12000*index,pt[0]:bottom_right[0]] = img_tmp[index][pt[1]:bottom_right[1],pt[0]:bottom_right[0]]

    return blackground_mask


def template_match(ori_img_path,template1_image,template2_image):
    ori_img = cv2.imread(ori_img_path)
    blackground_mask = np.zeros((ori_img.shape[0], ori_img.shape[1], 3), dtype="uint8")
    k = int(ori_img.shape[0]/15000)
    img_tmp = {}

    for index in range(k+1):
        # 定义一个数组保存图像
        if index == 0:
            img_tmp[index] = ori_img[0:15000*(index+1),0:ori_img.shape[1]]
        if index == k:
            img_tmp[index] = ori_img[12000*index:ori_img.shape[0],0:ori_img.shape[1]]
        else:
            img_tmp[index] = ori_img[12000*index:12000*index+15000,0:ori_img.shape[1]]
    threshold = 0.85
    res = {}
    loc = {}
    for index in range(k+1):
        res[index] =cv2.matchTemplate(img_tmp[index], template1_image, cv2.TM_CCOEFF_NORMED)
        loc[index] = np.where(res[index] >= threshold)
    res2 = {}
    loc2 = {}
    for index in range(k+1):
        res2[index] =cv2.matchTemplate(img_tmp[index], template2_image, cv2.TM_CCOEFF_NORMED)
        loc2[index] = np.where(res2[index] >= threshold)

    template_only_area = get_only_template_area(loc, blackground_mask,img_tmp,template1_image,k)

    template_only_area = get_only_template_area(loc2,template_only_area,img_tmp,template2_image,k)

    return template_only_area


def generate_img(template_only_area,pattern_path,folder_path):
    """生成想要的图片"""
    generate_folder_tmp = '/cloud_disk/users/huh/pcb/template_matching/NEW_TASK_4_6/test/'
    generate_folder_NO_RESIZE = osp.join(folder_path,'/segmap_layers/copper/',osp.basename(pattern_path))
    if not os.path.exists(generate_folder_NO_RESIZE):
        os.makedirs(generate_folder_NO_RESIZE)
    cv2.imwrite(generate_folder_NO_RESIZE,template_only_area)
    template_only_area =cv2.resize(template_only_area, (int(template_only_area.shape[0] / 60), int(template_only_area.shape[1] / 60)), cv2.INTER_NEAREST)
    test_path = osp.join(generate_folder_tmp,'test_'+osp.basename(pattern_path))
    cv2.imwrite(test_path,template_only_area)


def read_toml(toml_path):
    """读取toml文件中的数据"""
    with open(toml_path, 'r') as fp:
        content = load(fp)
    pattern_box = content['patterns']
    pattern0_box = pattern_box['pattern0']['bbox']
    pattern1_box = pattern_box['pattern0']['bbox']
    return pattern0_box,pattern1_box


def get_file_path(folder_path):
    """得到相关文件的路径"""
    file_list = os.listdir(folder_path)
    for file in file_list:
        if 'png' in file:
            ori_img_path = osp.join(folder_path,file)
        if 'toml' in file:
            toml_path = osp.join(folder_path,file)
    pattern0_folder_path = osp.join(folder_path+'/patterns/pattern0/copper/')
    pattern1_folder_path = osp.join(folder_path+'/patterns/pattern0/copper/')

    pattern0_path = osp.join(pattern0_folder_path,os.listdir(pattern0_folder_path)[0])
    pattern1_path = osp.join(pattern1_folder_path,os.listdir(pattern1_folder_path)[0])
    return pattern0_path,pattern1_path,toml_path,ori_img_path


def main():
    folder_path = '/cloud_disk/users/huh/pcb/template_matching/NEW_TASK_4_6/P80LI119982A0_L5/'
    pattern0_path,pattern1_path,toml_path,ori_img_path = get_file_path(folder_path)
    pattern0_bbox,pattern1_bbox = read_toml(toml_path)
    template1 = get_template(ori_img_path,pattern0_bbox)
    template2 = get_template(ori_img_path,pattern1_bbox)
    template_only_area = template_match(ori_img_path,template1,template2)
    generate_img(template_only_area,ori_img_path)


if __name__ == '__main__':
    main()

