"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/2/20 4:51 PM
"""

from tqdm import tqdm
import numpy as np
import cv2
from  graphcut import GraphMaker
lumda = 1
k = 1
def erode_dilate(img_path):
    """对copper进行腐蚀
    copper区域腐蚀一下，就是前景的hint
    背景区域腐蚀一下就是背景的hint
    """
    img_ori = cv2.imread(img_path)
    img_ori = img_ori*255
    # 开始进行腐蚀操作
    retVal, image = cv2.threshold(img_ori, 20, 255, cv2.THRESH_BINARY)
    corrosion_img = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))  ##腐蚀预处理，确定处理核的大小,矩阵操作
    pic_matrix = cv2.erode(image, corrosion_img, iterations=10)  # 进行腐蚀操作

    return pic_matrix


def get_new_img(img_end, save_p):
    cv2.imwrite(save_p, img_end)


def get_labels(background_hint,copper_hint):
    # 得到标签


    size = background_hint.shape[:2]
    labels = np.zeros(size)
    for i in tqdm(range((background_hint.shape[0]))):
        for j in range(background_hint.shape[1]):
            if background_hint[i,j][0] ==255:
                labels[i][j] = 1
                # background
                graphcut_class.add_seed(i,j,0)
    for i in tqdm(range(copper_hint.shape[0])):
        for j in range(copper_hint.shape[1]):
            if copper_hint[i, j][0] == 255:
                labels[i][j] = -1
                graphcut_class.add_seed(i, j, 1)
    return labels




if __name__ == '__main__':
    background_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/graphcut/sample1/1_background.png'
    copper_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/graphcut/sample1/1_copper.png'
    ori_img_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/graphcut/sample1/1_img.png'
    end_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/graphcut/test.jpg'
    im = cv2.imread(ori_img_path)

    # 初始化graphcut类
    graphcut_class = GraphMaker(ori_img_path,end_path)


    background_hint = erode_dilate(background_path)
    copper_hint = erode_dilate(copper_path)
    labels = get_labels(background_hint, copper_hint)

    graphcut_class.create_graph()
    graphcut_class.cut_graph()
    graphcut_class.save_image(end_path)




