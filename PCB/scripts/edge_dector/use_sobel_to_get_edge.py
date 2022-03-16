"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/14 11:01 AM
"""
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/pcb_small/20211022_10583-0-01-7_fake_B.png', 0)
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
#
# plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 3, 2), plt.imshow(sobelx, cmap='gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 3, 3), plt.imshow(sobely, cmap='gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.show()


"""canndy"""
import cv2
import numpy as np

# 以灰度图形式读入图像
img = cv2.imread('/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/pcb_small/20211022_10583-0-01-7_fake_B.png')
v1 = cv2.Canny(img, 80, 150, (3, 3))
v2 = cv2.Canny(img, 50, 100, (5, 5))

# np.vstack():在竖直方向上堆叠
# np.hstack():在水平方向上平铺堆叠
ret = np.hstack((v1, v2))
cv2.imshow('img', ret)
cv2.waitKey(0)
cv2.destroyAllWindows()