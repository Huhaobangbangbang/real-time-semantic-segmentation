"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/4/10 9:00 PM
"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np



def get_template(img_path):
    image = cv2.imread(img_path)
    # 获取白色区域的xy最大最小坐标
    # 得到cell的box边界
    where_255 = np.where(image == 255)
    listx = list(where_255)[0]
    listy = list(where_255)[1]
    x_min = listx[0]
    x_max = listx[-1]
    y_min = listy[0]
    y_max = listy[-1]
    template = image[x_min:x_max,y_min:y_max]

    cv2.imshow("img", template)
    cv2.waitKey(0)
    return template




if __name__ == '__main__':
    img_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/scripts/use_mask_match/get_template/pattern.png'
    template = get_template(img_path)
