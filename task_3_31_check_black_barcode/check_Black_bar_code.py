"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/31 6:25 PM
"""
import cv2
import numpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def non_max_suppression_fast(boxes, overlapThresh):
    """将矩形框中的矩形框去掉"""
    # 空数组检测
    if len(boxes) == 0:
        return []
    # 将类型转为float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    # 四个坐标数组
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算面积数组
    idxs = np.argsort(y2)  # 返回的是右下角坐标从小到大的索引值

    # 开始遍历删除重复的框
    while len(idxs) > 0:
        # 将最右下方的框放入pick数组
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # 找到剩下的其余框中最大的坐标x1y1，和最小的坐标x2y2,
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 计算重叠面积占对应框的比例
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        # 如果占比大于阈值，则删除
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


def get_word_area(image):
    """画出轮廓"""
    mser = cv2.MSER_create()

    vis = image.copy()
    regions, _ = mser.detectRegions(image)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    print(len(hulls))
    keep = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        keep.append([x, y, x + w, y + h])

    '''NMS是经常伴随图像区域检测的算法，作用是去除重复的区域，
        在人脸识别、物体检测等领域都经常使用，全称是非极大值抑制（non maximum suppression），
        就是抑制不是极大值的元素，所以用在这里就是抑制不是最大框的框，也就是去除大框中包含的小框'''
    # 使用NMS算法
    # keep2 = np.array(keep)
    # pick = non_max_suppression_fast(keep2, 0.5)
    # for (startX, startY, endX, endY) in pick:
    #     cv2.rectangle(vis, (startX, startY), (endX, endY), (255, 185, 120), 2)
    # 直接使用holyholes算法
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    return vis,len(hulls)


def erode_image(image):
    """开始进行腐蚀操作"""
    corrosion_img = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))  ##腐蚀预处理，确定处理核的大小,矩阵操作
    img3 = cv2.dilate(image, corrosion_img, iterations=10)  # 进行腐蚀操作
    return img3


def get_gray_scale(img_path):
    # 将图像灰度化
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 将图片二值化
    retVal, image = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)
    erode_image_ori = erode_image(image)
    vis,black_bar_number = get_word_area(erode_image_ori)

    cv2.imshow('grayimg', vis)
    cv2.waitKey(0)
    return black_bar_number


if __name__ == '__main__':
    # get the img_path
    sample_img_path = '/Users/huhao/Downloads/task_3_31/image/1.jpeg'
    black_bar_number = get_gray_scale(sample_img_path)
