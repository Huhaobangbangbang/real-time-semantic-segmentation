"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/6 3:40 PM
"""
import numpy as np
import os
# Much faster than the standard class
from fast_slic.avx2 import SlicAvx2
import maskslic as seg
import os.path as osp
from tqdm import tqdm
import cv2
from skimage import segmentation
"""超像素由一系列位置相邻且颜色、亮度、纹理等特征相似的像素点组成的小区域。
这些小区域大多保留了进一步进行图像分割的有效信息，且一般不会破坏图像中物体的边界信息，
用少量的超像素代替大量像素表达图像特征，降低了图像处理的复杂度，
一般作为分割算法的预处理步骤。"""

def get_ori_list(ori_folder):
    img_list = os.listdir(ori_folder)
    ori_list = []
    check_list = ['copper', 'bg', 'check', 'dust']
    for img_name in img_list:
        flag = 0
        for sample in check_list:
            if sample in img_name:
                flag = 1
                break
        if flag == 0:
            ori_list.append(osp.join(ori_folder, img_name))
        if len(ori_list) > 50:
            break
    return ori_list



def use_mask_to_get_two_areas(mask,end_result,image):
    """通过将指定mask来指定不画裂纹的区域"""
    where_0 = np.where(mask == False)
    end_result[where_0] = image[where_0]/255
    return end_result


def use_fast_slic_to_get_edge(img_path,mask_path,end_path):
    image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.imread(mask_path)
    img_R = mask_image[:, :, 0]
    mask = img_R > 220
    # use fast slic
    slic = SlicAvx2(num_components=8000, compactness=10)
    slic_result = slic.iterate(image, 10) # Cluster Map
    end_result = segmentation.mark_boundaries(image, slic_result, outline_color=(0, 1, 1))
    end_result = use_mask_to_get_two_areas(mask,end_result,image)


    cv2.imwrite(end_path,end_result*255)



if __name__ == '__main__':
    ori_folder = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/pcb_small'
    end_folder = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/slic_result'
    ori_list = get_ori_list(ori_folder)
    for img_path in tqdm(ori_list):
        end_path = osp.join(end_folder, osp.basename(img_path[:-4])+'_end_result.jpg')
        mask_path = osp.join(ori_folder, osp.basename(img_path[:-4]) + '_bg_mask.jpg')
        use_fast_slic_to_get_edge(img_path, mask_path, end_path)
        os.system("cp {} {}".format(img_path,end_folder))



