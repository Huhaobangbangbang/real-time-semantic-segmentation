"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/18 8:06 PM
"""
### 导入依赖包
import xml.etree.ElementTree as etree
from svgelements import *
import xml.etree.ElementTree as ET
# 打开文件
svg_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/scripts/template_matching/drawing.svg'
DOMTree = ET.parse(svg_path)
root = DOMTree.getroot()


for object in root.iter():
    # bndbox这部分看自己的标签是什么
    print('KKK')
    for sample in object.iter():
        print(sample)

    # # 读取xml文件中的object_name
    # object_name = object.iter('name')
    # print(object_name)
    # for bndbox in object_bndbox:
    #     node = []
    #     for child in bndbox:
    #         node.append(int(child.text))
    #     x, y = node[0], node[1]
    #     w, h = node[2], node[3]
    #
    # for name in object_name:
    #     name = name.text
    #     object_list.append("{} {} {} {} {}\n".format(x, y, w, h,name))


