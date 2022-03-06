"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/5 10:40 PM
"""
img_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/pcb_small/20211022_10583-0-01-7_fake_B.png'

import numpy as np

# Much faster than the standard class
from fast_slic.avx2 import SlicAvx2
from PIL import Image

with Image.open(img_path) as f:
   image = np.array(f)
# import cv2; image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)   # You can convert the image to CIELAB space if you need.
slic = SlicAvx2(num_components=1600, compactness=10)
assignment = slic.iterate(image) # Cluster Map
