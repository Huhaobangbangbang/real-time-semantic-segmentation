from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt
import cv2
from graphcut import GraphMaker
import numpy as np
from tqdm import tqdm
def get_seed(ori_img_path):
    """展示seed区域"""
    ori_image = cv2.imread(ori_img_path)
    #cv2.circle(ori_image, (245, 256), 3, [0, 255, 0], -1)
    #cv2.circle(ori_image, (245, 256), 3, [0, 255, 0], -1)
    #cv2.rectangle(ori_image, (240, 245),(250, 270), (0, 255, 0), 1)
    cv2.rectangle(ori_image, (265, 50), (500, 500), (0, 255, 0), 1)
    return ori_image


def fill_color_demo(ori_img_path):
    """使用floodfill算法来进行处理瑕疵"""
    image = cv2.imread(ori_img_path)
    flood_fill_img = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)  # mask必须行和列都加2，且必须为uint8单通道阵列
    # 为什么要加2可以这么理解：当从0行0列开始泛洪填充扫描时，mask多出来的2可以保证扫描的边界上的像素都会被处理
    cv2.floodFill(flood_fill_img, mask, (245, 256), (0, 255, 0), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)

    return  flood_fill_img


def erode_dilate(img_path):
    """对copper进行腐蚀
    copper区域腐蚀一下，就是前景的hint
    背景区域腐蚀一下就是背景的hint
    """
    img_ori = cv2.imread(img_path)
    # img_ori =  img_ori*255
    # 开始进行腐蚀操作
    retVal, image = cv2.threshold(img_ori, 20, 255, cv2.THRESH_BINARY)
    corrosion_img = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))  ##腐蚀预处理，确定处理核的大小,矩阵操作
    pic_matrix = cv2.erode(image, corrosion_img, iterations=10)  # 进行腐蚀操作

    return pic_matrix


def get_labels(copper_hint,foregraound_hint):
    # 得到标签
    for i in tqdm(range(240,250)):
        for j in range(245,270):
                # background 通过mask定义标签
                graphcut_class.add_seed(i, j, 1)


def graphcut(end_path):
    get_labels()
    graphcut_class.create_graph()
    graphcut_class.cut_graph()
    graphcut_class.save_image(end_path)


if __name__ == '__main__':
    # 20211022_10587-0-03-5_fake_B.png
    ori_img_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/pcb_small/20211022_10583-0-01-7_fake_B.png' # (245, 256)
    ori_image = get_seed(ori_img_path)
    #flood_fill_img = fill_color_demo(ori_img_path)
    # cv2.imshow("show the place which get seeds", ori_image)
    # if cv2.waitKey(0) == 9:
    #     cv2.destroyAllWindows()
    end_path = 'test.jpg'
    graphcut_class = GraphMaker(ori_img_path, end_path)
    graphcut(end_path)












    img_path_list = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/pcb_small'

