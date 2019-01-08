

import os
from PIL import Image
import glob

"""
0.2665253102535907 
31.03736755219588 116.45185788420909 
785 32
"""

def get_image_average_hw(image_path):

    addrs_image = glob.glob(image_path+'/*.jpg')
    image_num = len(addrs_image)

    width=0
    hight=0
    max_hight=0
    max_width=0
    i=0
    for img_addr in addrs_image:
        print(i)
        i = i+1
        im = Image.open(img_addr)
        w,h=im.size
        width = width+w
        hight = hight + h
        if w>max_width:
            max_width = w
        if h>max_hight:
            max_hight=h
    print(hight/width,hight/image_num,width/image_num,max_width,max_hight)
if __name__ == '__main__':
    get_image_average_hw('synth90k_million')











# import os
# from PIL import Image
# import glob
#
# path_file = "../data130/ocr/synth90k/imlist_full.txt"
#
# def get_image_path(path_file):
#     img_path_list = []
#     with open(path_file,"r") as fread:
#         for line in fread:
#             img_path_list.append(line)
#     return img_path_list
#
#
#
#
#
# def get_image_average_hw():
#
#
#     addrs_image = get_image_path(path_file)
#
#     width=0
#     hight=0
#     max_hight=0
#     max_width=0
#     i=0
#     for img_addr in addrs_image:
#         print(i)
#         i = i+1
#         im = Image.open(img_addr)
#         w,h=im.size
#         width = width+w
#         hight = hight + h
#         if w>max_width:
#             max_width = w
#         if h>max_hight:
#             max_hight=h
#     print(hight/width,max_width,max_hight)
# if __name__ == '__main__':
#     get_image_average_hw()