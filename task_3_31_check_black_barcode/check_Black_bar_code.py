"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/31 6:25 PM
"""
import cv2
import numpy as np


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
    """得到轮廓数量"""
    mser = cv2.MSER_create()

    vis = image.copy()
    regions, _ = mser.detectRegions(image)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    # cv2.imshow("img", vis)
    # cv2.waitKey(0)

    return vis,len(hulls),hulls


def erode_image(image):
    """开始进行腐蚀操作"""
    corrosion_img = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))  ##腐蚀预处理，确定处理核的大小,矩阵操作
    img3 = cv2.dilate(image, corrosion_img, iterations=10)  # 进行腐蚀操作
    return img3

def delicate_img(image):
    """ 膨胀"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    expand_pic = cv2.dilate(image, kernel)
    return expand_pic

def show_box(hulls,sample_img_path):
    image = cv2.imread(sample_img_path)
    min_y_index = 10000
    max_y = -1
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        if y<min_y_index:
            min_y_index = y
            min_y = y+h
        if y>max_y:
            max_y =y
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image,min_y,max_y



def process_left_img(image,min_y,max_y,black_bar_number):
    """得到左边部分黑色条形码数量"""
    left_img = image[0:image.shape[0], 0:int((min_y + max_y) / 2)]

    _, left_black_bar_number, _ = get_word_area(left_img)

    if left_black_bar_number ==0:
        left_img = image[0:image.shape[0], 0:max_y + 50]
        _, left_black_bar_number, _ = get_word_area(left_img)
    if left_black_bar_number ==black_bar_number:
        left_img = image[0:image.shape[0], 0:max_y -35]
        _, left_black_bar_number, _ = get_word_area(left_img)

    # cv2.imshow("result img", left_img)
    # cv2.waitKey(0)
    return left_black_bar_number


def get_gray_scale(img_path):
    # 将图像灰度化
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 将图片二值化
    retVal, image = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    erode_image_ori = erode_image(image)
    # 得到所有的黑色条码数量
    vis,black_bar_number,hulls = get_word_area(erode_image_ori)
    cv2.imshow("result img", vis)
    cv2.waitKey(0)
    image,min_y,max_y = show_box(hulls, img_path)
    left_black_bar_number = process_left_img(vis,min_y,max_y,black_bar_number)

    # cv2.imshow("result img", image)
    # cv2.waitKey(0)
    return black_bar_number,left_black_bar_number,black_bar_number -left_black_bar_number


if __name__ == '__main__':
    # get the img_path
    sample_img_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/task_3_31_check_black_barcode/image/20.jpeg'
    black_bar_number,left_black_bar_number,right_black_bar_number = get_gray_scale(sample_img_path)
    print('左边的黑色条码数为', left_black_bar_number)
    print('右边的黑色条码数为', right_black_bar_number)
    print('所有条码的个数',black_bar_number)
