import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import os.path as osp
def erode_dilate(img_ori):
    """对copper进行腐蚀
    copper区域腐蚀一下，就是前景的hint
    背景区域腐蚀一下就是背景的hint
    """

    # img_ori =  img_ori*255
    # 开始进行腐蚀操作
    retVal, image = cv2.threshold(img_ori, 20, 255, cv2.THRESH_BINARY)
    corrosion_img = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))  ##腐蚀预处理，确定处理核的大小,矩阵操作
    pic_matrix = cv2.erode(image, corrosion_img, iterations=10)  # 进行腐蚀操作

    return pic_matrix


def get_ori_list(ori_folder):
    img_list = os.listdir(ori_folder)
    ori_list = []
    check_list = ['copper','bg','check','dust']
    for img_name in img_list:
        flag = 0
        for sample in check_list:
            if sample in img_name:
                flag=1
                break
        if flag==0:
            ori_list.append(osp.join(ori_folder,img_name))
        if len(ori_list)>1:
            break
    return ori_list

def get_new_markers(markers,background_hint,fore_hint):
    for i in tqdm(range(fore_hint.shape[0])):
        for j in range(fore_hint.shape[1]):
            if fore_hint[i, j][0] == 255:
                markers[i][j] = 2
    for i in tqdm(range(background_hint.shape[0])):
        for j in range(background_hint.shape[1]):
            if background_hint[i, j][0] == 255:
                # foreground 通过mask定义标签
                markers[i][j] = 1
    return markers

def watershed(image_path,background_hint,fore_hint):
    image = cv2.imread(image_path)
    # 第一步：灰度处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 第二步：二值化处理 + 反色处理 255 -> 0 | 0 -> 255
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # 通过距离变换的结果取二值化，得到前景
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(),
                                 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = get_new_markers(markers, background_hint, fore_hint)

    markers[unknown == 255] = 0
    sure_bg[unknown == 255] = 0
    sure_bg[sure_bg == 255] = 2
    sure_bg = sure_bg.astype(np.int32)
    # 分水岭只是对0的位置进行分割 1-背景 0-待分割 2-前景
    result = cv2.watershed(image, markers=sure_bg)  # 分水岭只是对0的位置进行分割 1-背景 0-待分割 2-前景
    image[result == -1] = [255, 0, 0]  # 分水岭标记为红色
    cv2.imwrite('watershed_res.jpg',image)
    # plt.figure(0)
    #
    # plt.subplot(231)
    # plt.title("binary")
    # plt.imshow(binary)
    #
    # plt.subplot(232)
    # plt.title("seed new")
    # plt.imshow(sure_bg)
    #
    # plt.subplot(233)
    # plt.title("distance")
    # plt.imshow(dist_transform * 50)
    #
    # plt.subplot(234)
    # plt.title("seed ori")
    # plt.imshow(markers)
    #
    # plt.subplot(235)
    # plt.title("result markers")
    # plt.imshow(result)
    #
    # plt.subplot(236)
    # plt.title("watershed")
    # plt.imshow(image)
    #
    # # plt.savefig(os.path.join(output_dir, "watershed_" + image_name))
    # plt.show()
    #

    # end_folder = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/scripts/watershed_result'
    # end_path = osp.join(end_folder,osp.basename(image_path))
    # cv2.imwrite(end_path,image)

def get_background(background_img_path,fore_img_path):
    background_hint = cv2.imread(background_img_path)
    fore_hint = cv2.imread(fore_img_path)

    b_where_255 = np.where(background_hint == 255)
    f_where_255 = np.where(fore_hint == 255)
    background_matrix = np.zeros((512, 512, 3), dtype = "uint8")
    fore_matrix  = np.zeros((512, 512, 3), dtype = "uint8")
    background_matrix[b_where_255]=255
    background_matrix[f_where_255] = 255
    background_where_0 = np.where(background_matrix == 0)
    fore_matrix[background_where_0] = 255
    return background_matrix,fore_matrix


if __name__ == '__main__':
    ori_folder = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/pcb_small'
    ori_list = get_ori_list(ori_folder)
    # for img_path in ori_list:
    #     background_img_path = osp.join(ori_folder,osp.basename(img_path[:-4])+'_bg_mask.jpg')
    #     fore_img_path = osp.join(ori_folder,osp.basename(img_path[:-4])+'_cooper_mask.jpg')
    #     background_matrix, fore_matrix = get_background(background_img_path,fore_img_path)
    #     background_hint = erode_dilate(background_matrix)
    #     fore_hint = erode_dilate(fore_matrix)
    #     watershed(img_path,background_hint,fore_hint)
    img_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/pcb_small/20211022_10583-0-01-7_fake_B.png'
    background_img_path = osp.join(ori_folder, osp.basename(img_path[:-4]) + '_bg_mask.jpg')
    fore_img_path = osp.join(ori_folder,osp.basename(img_path[:-4])+'_copper_mask.jpg')
    background_matrix, fore_matrix = get_background(background_img_path,fore_img_path)
    background_hint = erode_dilate(background_matrix)
    fore_hint = erode_dilate(fore_matrix)
    watershed(img_path,background_hint,fore_hint)
