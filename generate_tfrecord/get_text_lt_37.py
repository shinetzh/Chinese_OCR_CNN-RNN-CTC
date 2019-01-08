
import os
from PIL import Image
import glob



def get_text_dir(img_dir,text_fold):
    img_name_head = img_dir.find("/") + 1
    text_title = img_dir[img_name_head:-4] + '.txt'
    text_dir = os.path.join(text_fold, text_title)
    return text_dir

def get_img_dir(text_dir,img_fold):
    text_name_head = text_dir.find("/") + 1
    img_title = text_dir[text_name_head:-4] + '.jpg'
    img_dir = os.path.join(img_fold, img_title)
    return img_dir

def del_image_label(text_path,img_path):

    addrs_text = glob.glob(text_path+'/*.txt')

    num_text_lt_37 = 0

    for text_addr in addrs_text:
        with open(text_addr, "r", encoding="utf-8") as f_r:
            text = f_r.read()
            if len(text)<37:
                num_text_lt_37=num_text_lt_37+1
                continue
            else:
                os.remove(text_addr)
                print(text_addr + " deleted")
                img_dir = get_img_dir(text_addr, img_path)
                if os.path.exists(img_dir):
                    os.remove(img_dir)
    print(num_text_lt_37)

if __name__ == '__main__':
    del_image_label('train_crop_annot_37','train_crop_37')