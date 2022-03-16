"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/16 11:31 AM
"""
import cv2
from tqdm import tqdm
import os.path as osp
import numpy as np

def get_background(background_img_path,fore_img_path):
    """得到背景和前景区域"""
    background_hint = cv2.imread(background_img_path)
    fore_hint = cv2.imread(fore_img_path)
    background_hint = erode_dilate(background_hint)
    fore_hint = erode_dilate(fore_hint)


    b_where_255 = np.where(background_hint == 255)
    f_where_255 = np.where(fore_hint == 255)
    background_matrix = np.zeros((512, 512, 3), dtype = "uint8")
    fore_matrix  = np.zeros((512, 512, 3), dtype = "uint8")
    background_matrix[b_where_255]=255
    background_matrix[f_where_255] = 255
    background_where_0 = np.where(background_matrix == 0)
    fore_matrix[background_where_0] = 255

    return background_matrix,fore_matrix


def erode_dilate(img_ori):
    """腐蚀"""
    # 开始进行腐蚀操作
    retVal, image = cv2.threshold(img_ori, 200, 255, cv2.THRESH_BINARY)
    corrosion_img = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))  ##腐蚀预处理，确定处理核的大小,矩阵操作
    pic_matrix = cv2.erode(image, corrosion_img, iterations=10)  # 进行腐蚀操作

    return pic_matrix

def expand(img_ori):
    """膨胀"""
    # 开始进行腐蚀操作
    retVal, image = cv2.threshold(img_ori, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    expand_pic = cv2.dilate(image, kernel)
    return expand_pic

def get_new_markers(markers,background_hint,fore_hint):
    for i in tqdm(range(background_hint.shape[0])):
        for j in range(background_hint.shape[1]):
            if background_hint[i, j][0] == 255:
                # foreground 通过mask定义标签
                markers[i][j] = 1
    for i in tqdm(range(fore_hint.shape[0])):
        for j in range(fore_hint.shape[1]):
            if fore_hint[i, j][0] == 255:
                markers[i][j] = 2

    return markers

def use_watershed_get_edge(img_path, background_hint, fore_hint):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Step2.阈值分割，将图像分为黑白两部分
    ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Step3. 对图像进行“开运算”，先腐蚀再膨胀
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=10)
    # Step4. 对“开运算”的结果进行膨胀，得到大部分都是背景的区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Step5.通过distanceTransform获取前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DIST_L1 DIST_C只能 对应掩膜为3    DIST_L2 可以为3或者5
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    # Step6. sure_bg与sure_fg相减,得到既有前景又有背景的重合区域   #此区域和轮廓区域的关系未知
    sure_fg = np.uint8(sure_fg)
    unknow = cv2.subtract(sure_bg, sure_fg)
    # Step7. 连通区域处理
    ret, markers = cv2.connectedComponents(sure_fg, connectivity=8)  # 对连通区域进行标号  序号为 0 - N-1
    #markers = markers + 1
    #markers[unknow == 255] = 0
    markers = get_new_markers(markers, background_hint, fore_hint)
    # 分水岭算法
    markers = cv2.watershed(img, markers)  # 分水岭算法后，所有轮廓的像素点被标注为  -1
    img[markers == -1] = [0, 0, 255]  # 标注为-1 的像素点标 红

    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ori_folder = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/pcb_small'
    img_name = '20211022_10583-0-01-7_fake_B.png'
    img_path = osp.join(ori_folder,img_name)
    background_img_path = osp.join(ori_folder, osp.basename(img_path[:-4]) + '_bg_mask.jpg')
    fore_img_path = osp.join(ori_folder, osp.basename(img_path[:-4]) + '_copper_mask.jpg')
    background_hint, fore_hint = get_background(background_img_path, fore_img_path)

    # background_hint = erode_dilate(background_matrix)
    # fore_hint = erode_dilate(fore_matrix)


    use_watershed_get_edge(img_path, background_hint, fore_hint)