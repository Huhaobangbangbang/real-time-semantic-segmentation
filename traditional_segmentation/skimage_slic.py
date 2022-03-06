"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/4 9:32 AM
"""
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

#加载图片
image = img_as_float(io.imread("1.jpg"))

#
for numSegments in (100, 200, 300):
    #SLIC
    segments = slic(image, n_segments = numSegments, sigma = 5)

    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")

# show the plots
plt.show()