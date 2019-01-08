import sys
import random
import os
import glob
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
import numpy as np

# text_fold = 'train_crop_annot_37'
text_fold = 'train_crop_annot'
pixel_per_char = 512
tarShape = (300, 250)



def get_text_dir(img_dir,text_fold):
    img_name = img_dir.split('/')[-1]
    text_title = img_name[:-3] + 'txt'
    text_dir = os.path.join(text_fold, text_title)
    return text_title,text_dir



def pad_image(image, target_size):
    iw, ih = image.size  # source size
    w, h = target_size  # target size
    nw = iw
    nh = ih
    if iw>w or ih>h:
        scale = min(w / iw, h / ih)  # minimal ratio


        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128, 128, 128))  # new gray image
    # // round down to int
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # fill the center of new gray image with target image

    return new_image



def is_every_char_big_enough(shape,img_path,pixel_per_char):
    area =shape[0]*shape[1]
    _, text_dir = get_text_dir(img_path, text_fold)
    with open(text_dir, "r", encoding="utf-8") as f_r:
        text = f_r.read()
        area_per_char = area/len(text)
        if area_per_char<pixel_per_char:
            return False
        else:
            return True

def get_copy_file(fileDir,file_num):
    if not os.path.exists(fileDir):
        print("the fileDir doesn't exist")
        return

    pathDir = glob.glob(fileDir+'/*.jpg')
    num_all = len(pathDir)
    print(num_all)
    truncation_probability=float(file_num)/num_all
    sample=[]
    cur_num=0
    for file in pathDir:
        print(cur_num)
        img = Image.open(file)
        img_array = np.array(img)
        # print(img_array.shape)
        if len(img_array.shape)<3 or img_array.shape[2]<3 or \
                (not is_every_char_big_enough(img_array.shape,file,pixel_per_char)):
            continue
        if(random.randint(0,1)<truncation_probability):
            sample.append(file)
            cur_num=cur_num+1
            if(cur_num>=int(file_num)):
                break

    return sample

# resize and store image with tensorflow methods

# def resize_and_store_image(sample_L,tarShape,tarDir):
#     if not os.path.exists(tarDir):
#         os.makedirs(tarDir)
#
#     with tf.Session() as sess:
#         cur=1
#         for img_dir in sample_L:
#             # copy correspoding text to tarDir
#             text_title, text_dir = get_text_dir(img_dir, text_fold)
#             shutil.copy(text_dir,os.path.join(tarDir,text_title))
#
#
#             print(('{}/{}'+img_dir).format(cur,len(sample_L)))
#             image_raw_data = tf.gfile.FastGFile(img_dir,'rb').read()
#             img_data = tf.image.decode_jpeg(image_raw_data)
#
#             img_resized = tf.image.resize_images(img_data,tarShape,method=1)
#
#             # channel_num = img_resized.eval().shape[2]
#             # noise_img = generate_noise_image((150, 450, channel_num))
#             # noise_img = tf.image.convert_image_dtype(noise_img, dtype=tf.uint8)
#             # img_resized = tf.concat([img_resized, noise_img], 1)
#
#             img_resized = tf.image.convert_image_dtype(img_resized,dtype=tf.uint8)
#             print(type(img_resized))
#             encoded_image = tf.image.encode_jpeg(img_resized)
#             with tf.gfile.GFile(os.path.join(tarDir,img_dir.split('/')[-1]),"wb") as f:
#                 f.write(encoded_image.eval())
#             cur = cur+1


def resize_and_store_image(sample_L,tarShape,tarDir):
    if not os.path.exists(tarDir):
            os.makedirs(tarDir)
    cur = 1
    for img_dir in sample_L:
        text_title, text_dir = get_text_dir(img_dir, text_fold)
        shutil.copy(text_dir,os.path.join(tarDir,text_title))


        print(('{}/{}'+img_dir).format(cur,len(sample_L)))
        img = Image.open(img_dir)
        # img_resized = pad_image(img, tarShape)
        img.save(os.path.join(tarDir,img_dir.split('/')[-1]))
        cur = cur + 1


def generate_noise_image(shape):
    out_img=tf.random_uniform(shape,0,255)
    return out_img


def main(argv):
    if(len(argv)<4):
        print('need parameters: fileDir,file_num,tarDir')
        return
    sample = get_copy_file(argv[1],argv[2])
    resize_and_store_image(sample,tarShape,argv[3])

if __name__ == "__main__":
    main(sys.argv)