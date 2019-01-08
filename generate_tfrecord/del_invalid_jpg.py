
import os
from PIL import Image
import glob



def get_text_dir(img_dir,text_fold):
    img_name_head = img_dir.find("/") + 1
    text_title = img_dir[img_name_head:-4] + '.txt'
    text_dir = os.path.join(text_fold, text_title)
    return text_dir

def del_image_label(image_path):

    addrs_image = glob.glob(image_path+'/*.jpg')

    #label_path = './train100/*.txt'
    # addrs_label = glob.glob(label_path+'/*.txt')

    invalid_jpg_num = 0

    for img_addr in addrs_image:
        try:
            im = Image.open(img_addr)
            # do stuff
        except IOError:
            invalid_jpg_num = invalid_jpg_num + 1
            os.remove(img_addr)
            print(img_addr + " deleted")
            # text_dir = get_text_dir(img_addr,label_path)
            # os.remove(text_dir)
            # print(text_dir + " deleted")
    print(invalid_jpg_num)

if __name__ == '__main__':

    del_image_label('synth90k_million')