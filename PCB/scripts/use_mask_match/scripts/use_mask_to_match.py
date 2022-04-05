from re import template
import cv2
from matplotlib import image
import numpy as np
from tqdm import tqdm
import os.path as osp
import os
def template_match(ori_img_path,template):
    ori_img = cv2.imread(ori_img_path)
    h, w = template.shape[:2]
    img_gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    # 取匹配程度大于%90的坐标
    loc = np.where(res >= threshold)
    # np.where返回的坐标值(x,y)是(h,w)，注意h,w的顺序
    index = 0
    for pt in zip(*loc[::-1]):
        index = index + 1

        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(ori_img, pt, bottom_right, (0, 255, 0), 10)
    # cv2.imwrite("img.jpg", img_rgb)
    #cv2.rectangle(ori_img, (1139,3456) ,(4620,5947), (0,255,0), 10)
    end_img_path = osp.join('/Users/huhao/Downloads/','KKK'+osp.basename(ori_img_path))
    cv2.imwrite(end_img_path, ori_img)




def get_points(border_mask_path):
    """通过border_mask来得到模板的坐标"""
    pic_matrix = cv2.imread(border_mask_path)

    border_pos = []
    for i in tqdm(range(int(pic_matrix.shape[0]/2))):
        for j in range(int(pic_matrix.shape[1])):
            if pic_matrix[i][j][0] == 255:
                border_pos.append([i,j])
            if len(border_pos) == 1:
                break
        if len(border_pos) == 1:
            break
    num_x = int(pic_matrix.shape[0])-1
    num_y = int(pic_matrix.shape[1])-1
    for i in tqdm(range(4500,num_x)):
        for j in range(num_y):
            if pic_matrix[i][num_y-j][0] == 255:
                border_pos.append([i,num_y-j-1])
            if len(border_pos) == 2:
                break
        if len(border_pos) == 2:
            break

    return border_pos


def get_template(border_path,border_pos):
    image = cv2.imread(border_path)
    template = image[border_pos[0][0]:border_pos[1][0],border_pos[0][1]:border_pos[1][1]]
    cv2.imwrite('/Users/huhao/Downloads/gray_template/0969P4I63661A0_L2_template.jpg', template)

def show_template(border_path):
    image = cv2.imread(border_path)
    cv2.rectangle(image,(border_pos[0][1],border_pos[0][0]),(border_pos[1][1],border_pos[1][0]),(0,255,0),10)
    cv2.imwrite('/Users/huhao/Downloads/show_template.jpg', image)


if __name__ == '__main__':
    #border_mask_path = '/Users/huhao/Downloads/Template_img/0_ok/PZ12C19248A0_L5/annotations/patterns/pattern0/border_points/20211024T200738S690_cam1_7783.png'
    #border_path = '/Users/huhao/Downloads/Template_img/0_ok/PZ12C19248A0_L5/annotations/patterns/pattern1/copper/20211024T200738S690_cam1_7783.png'
    border_mask_path = '/Users/huhao/Downloads/Template_img/0_ok/0969P4I63661A0_L2/annotations/patterns/pattern0/border_points/20211024T022047S581_cam10_1177.png'
    border_path = '/Users/huhao/Downloads/Template_img/0_ok/0969P4I63661A0_L2/annotations/patterns/pattern0/copper/20211024T022047S581_cam10_1177.png'
    border_pos = get_points(border_mask_path)

    get_template(border_path, border_pos)




