"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/4/8 9:11 AM
"""
import os.path as osp
import os
from re import template

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
from matplotlib import image
import numpy as np
from tqdm import tqdm


def get_template(img_path, pattern0_bbox):
    """通过pattern0_bbox得到位置信息"""
    image = cv2.imread(img_path)
    template = image[pattern0_bbox[1]:pattern0_bbox[3], pattern0_bbox[0]:pattern0_bbox[2]]

    return template


def test(folder_path):
    """检查所有的图片，看看哪一张有框"""
    ori_folder = '/cloud_disk/users/huh/pcb/template_matching/Template_img/Template_CAD/'
    file_list = os.listdir(ori_folder)
    for img in tqdm(file_list):
        ori_img_path = osp.join(ori_folder, img)
        image = template_match(ori_img_path, template1, template2)
        re = cv2.resize(image, (int(image.shape[0] / 60), int(image.shape[1] / 60)), cv2.INTER_NEAREST)
        cv2.imwrite('/cloud_disk/users/huh/pcb/template_matching/NEW_TASK_4_6/test/' + img, re)


def use_loc_to_rectangle(loc1, loc2, loc3, ori_img, template_image, ori_img1, ori_img2, ori_img3):
    h, w = template_image.shape[:2]
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


def get_only_template_area(loc1, loc2, loc3, blackground_mask, ori_img1, ori_img2, ori_img3, template_image):
    h, w = template_image.shape[:2]
    for pt in zip(*loc1[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        blackground_mask[pt[1]:bottom_right[1], pt[0]:bottom_right[0]] = ori_img1[pt[1]:bottom_right[1],
                                                                         pt[0]:bottom_right[0]]
    for pt in zip(*loc2[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        blackground_mask[pt[1]:bottom_right[1], pt[0] + 12000:bottom_right[0] + 12000] = ori_img2[pt[1]:bottom_right[1],
                                                                                         pt[0]:bottom_right[0]]

    for pt in zip(*loc3[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        print(blackground_mask[pt[1]:bottom_right[1], pt[0] + 24000:bottom_right[0] + 24000].shape)
        print(ori_img2[pt[1]:bottom_right[1], pt[0]:bottom_right[0]].shape)
        blackground_mask[pt[1]:bottom_right[1], pt[0] + 24000:bottom_right[0] + 24000] = ori_img3[pt[1]:bottom_right[1],
                                                                                         pt[0]:bottom_right[0]]

    return blackground_mask


def get_rectangle_img(loc1, loc2, loc3, ori_img, template1_image, ori_img1, ori_img2, ori_img3, loc4, loc5, loc6,
                      template2_image):
    """得到画了框框的图像"""
    image = use_loc_to_rectangle(loc1, loc2, loc3, ori_img, template1_image, ori_img1, ori_img2, ori_img3)
    ori_img4 = image[0:image.shape[0], 0:16000]
    ori_img5 = image[0:image.shape[0], 12000:28000]
    ori_img6 = image[0:image.shape[0], 24000:image.shape[1]]
    ori_image_end = use_loc_to_rectangle(loc4, loc5, loc6, image, template2_image, ori_img4, ori_img5, ori_img6)
    return ori_image_end


def template_match(ori_img_path, template1_image, template2_image):
    ori_img = cv2.imread(ori_img_path)
    blackground_mask = np.zeros((ori_img.shape[0], ori_img.shape[1], 3), dtype="uint8")
    ori_img1 = ori_img[0:ori_img.shape[0], 0:16000]
    ori_img2 = ori_img[0:ori_img.shape[0], 12000:28000]
    ori_img3 = ori_img[0:ori_img.shape[0], 24000:ori_img.shape[1]]
    threshold = 0.85
    res1 = cv2.matchTemplate(ori_img1, template1_image, cv2.TM_CCOEFF_NORMED)
    loc1 = np.where(res1 >= threshold)
    res2 = cv2.matchTemplate(ori_img2, template1_image, cv2.TM_CCOEFF_NORMED)
    loc2 = np.where(res2 >= threshold)
    res3 = cv2.matchTemplate(ori_img3, template1_image, cv2.TM_CCOEFF_NORMED)
    loc3 = np.where(res3 >= threshold)
    res4 = cv2.matchTemplate(ori_img1, template2_image, cv2.TM_CCOEFF_NORMED)
    loc4 = np.where(res4 >= threshold)
    res5 = cv2.matchTemplate(ori_img2, template2_image, cv2.TM_CCOEFF_NORMED)
    loc5 = np.where(res5 >= threshold)
    res6 = cv2.matchTemplate(ori_img3, template2_image, cv2.TM_CCOEFF_NORMED)
    loc6 = np.where(res6 >= threshold)
    # ori_image_end = get_rectangle_img(loc1,loc2,loc3,ori_img,template1_image,ori_img1,ori_img2,ori_img3,loc4,loc5,loc6,template2_image)
    # 得到只包含模板的mask
    template_only_area = get_only_template_area(loc1, loc2, loc3, blackground_mask, ori_img1, ori_img2, ori_img3,
                                                template1_image)
    template_only_area = get_only_template_area(loc4, loc5, loc6, template_only_area, ori_img1, ori_img2, ori_img3,
                                                template2_image)

    return template_only_area


if __name__ == '__main__':
    pattern1_path = '/cloud_disk/users/huh/pcb/template_matching/NEW_TASK_4_6/PZ12C19248A0_L4/patterns/pattern0/copper/pattern0.png'
    pattern2_path = '/cloud_disk/users/huh/pcb/template_matching/NEW_TASK_4_6/PZ12C19248A0_L4/patterns/pattern1/copper/pattern1.png'
    pattern0_bbox = [2316, 7551, 4106, 8783]
    pattern1_bbox = [2316, 8907, 4106, 10155]
    template1 = get_template(pattern1_path, pattern0_bbox)
    template2 = get_template(pattern2_path, pattern1_bbox)
    # template =cv2.resize(template1, (int(template1.shape[0] / 5), int(template1.shape[1] / 5)), cv2.INTER_NEAREST)
    # cv2.imwrite('/cloud_disk/users/huh/pcb/template_matching/NEW_TASK_4_6/scripts/test1.jpg',template)
    # template =cv2.resize(template2, (int(template2.shape[0] / 5), int(template2.shape[1] / 5)), cv2.INTER_NEAREST)
    # cv2.imwrite('/cloud_disk/users/huh/pcb/template_matching/NEW_TASK_4_6/scripts/test2.jpg',template)

    img_path = '/cloud_disk/users/huh/pcb/template_matching/Template_img/Template_CAD/PZ12C19248A0_L5-211023_7783A.png'
    template_only_area = template_match(img_path, template1, template2)

    template_only_area = cv2.resize(template_only_area,
                                    (int(template_only_area.shape[0] / 50), int(template_only_area.shape[1] / 50)),
                                    cv2.INTER_NEAREST)
    cv2.imwrite('/cloud_disk/users/huh/pcb/template_matching/NEW_TASK_4_6/scripts/test.jpg', template_only_area)







