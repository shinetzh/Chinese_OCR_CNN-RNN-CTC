
import tensorflow as tf
import glob
from scipy.misc import imread, imresize, imsave
from PIL import Image,ImageFile
import numpy as np
import sys
import io
import os
import re

# all:995865
# use:500000
# resized_images:576566

num_files = 10
num_instance = 10000
tar_shape=(200,60)
padding_lenth = 24
null_code = 133
img_save_fold = 'resized_image_2'

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_charset(filename, null_character=u'\u2591'):
  """Reads a charset definition from a tab separated text file.

  charset file has to have format compatible with the FSNS dataset.

  Args:
    filename: a path to the charset file.
    null_character: a unicode character used to replace '<null>' character. the
      default value is a light shade block '░'.

  Returns:
    a dictionary with keys equal to character codes and values - unicode
    characters.
  """
  pattern = re.compile(r'(\d+)\t(.+)')
  charset = {}
  with tf.gfile.GFile(filename) as f:
    for i, line in enumerate(f):
      m = pattern.match(line)
      if m is None:
        print('incorrect charset file. line #{}: {}', i, line)
        continue
      code = int(m.group(1))
      char = m.group(2)
      if char == '<nul>':
        char = null_character
      charset[code] = char
  return charset




def encode_utf8_string(text, length, dic, null_char_id=5000):
    char_ids_padded = [null_char_id]*length
    char_ids_unpadded = [null_char_id]*len(text)
    for i in range(len(text)):
        #print(i,text[i])
        hash_id = dic[text[i]]
        char_ids_padded[i] = hash_id
        char_ids_unpadded[i] = hash_id
    return char_ids_padded, char_ids_unpadded



def get_text_dir(img_dir,text_fold):
    img_name_head = img_dir.find("/") + 1
    text_title = img_dir[img_name_head:-4] + '.txt'
    text_dir = os.path.join(text_fold, text_title)
    return text_dir


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




def main(argv):
    if(len(argv)<4):
        print("please input args(image_path,dict_path,train/test/val)")
        return
    image_path = argv[1]
    dict_path = argv[2]
    data_for = argv[3]
    print(image_path,dict_path,data_for)
    #image_path = './train100/*.jpg'
    addrs_image = glob.glob(image_path+'/*.jpg')

    print("produce dict")
    dic = read_charset(dict_path, null_character=u'\u2591')
    dic = dict(zip(dic.values(),dic.keys()))
    print(dic)
    # dic={}
    # with open(dict_path, encoding="utf-8") as dict_file:
    #     for line in dict_file:
    #         print(line)
    #         (key, value) = line.strip().split('\t')
    #         dic[value] = int(key)
    # dic[" "]=0
    # print(len(dic))
    # print(dic)

    print(len(addrs_image))



    for i in range(num_files):
        print("---------------------------------------------------------------------------------------------------------write ",i," file")
        fileName=(data_for+"-tfrecords-%.5d-of-%.5d" % (i,num_files))
        if not os.path.exists(os.path.join('tfrecord_crnn',data_for)):
            os.makedirs(os.path.join('tfrecord_crnn',data_for))
        tfrecord_writer  = tf.python_io.TFRecordWriter(path='tfrecord_crnn/'+data_for+'/'+fileName)
        for j in range(num_instance*i,num_instance*(i+1)):

            print(addrs_image[j])

            image = Image.open(addrs_image[j])
            image = image.convert('L')
            print(image.size)
            img = pad_image(image, tar_shape)
            img.save(os.path.join(img_save_fold, addrs_image[j].split('/')[-1]))
            img_array = np.array(img)

            # store img as JPEG bytes
            img_bytes = io.BytesIO()
            imsave(img_bytes, img, format='JPEG')
            img_bytes.seek(0)

            #get image text dir

            text = addrs_image[j].split('_')[2]
            #tracking the progress
            print(str(j+1)+'/'+str(len(addrs_image))+" "+text)

            char_ids_padded, char_ids_unpadded = encode_utf8_string(
                text, padding_lenth, dic, null_code)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image/format': _bytes_feature(b"jpg"),
                    'image/encoded': _bytes_feature(img_bytes.getvalue()),
                    'image/class': _int64_feature(char_ids_padded),
                    'image/unpadded_class': _int64_feature(char_ids_unpadded),
                    'height': _int64_feature([img_array.shape[0]]),
                    'width': _int64_feature([img_array.shape[1]]),
                    'orig_width': _int64_feature([img_array.shape[1]]),
                    'image/text': _bytes_feature(bytes(text, 'utf-8'))
                }
            ))
            tfrecord_writer.write(example.SerializeToString())
        tfrecord_writer.close()



if __name__ == '__main__':
    main(sys.argv)




