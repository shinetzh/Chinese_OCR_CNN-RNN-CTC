import json
import os

json_file="crop_annot.json"
jsondata = json.load(open(json_file, 'r'))

imagelist=jsondata.keys()

text_dir = 'crop_coco_train_annot'

if not os.path.exists(text_dir):
    os.makedirs(text_dir)

for image_name in imagelist:
    text_title=image_name[0:-4]+'.txt'
    image_text=open(os.path.join(text_dir,text_title),'w')
    print(image_name)
    print(len(jsondata[image_name]['annotations']))
    for annot_index in range(len(jsondata[image_name]['annotations'])):
        str = jsondata[image_name]['annotations'][annot_index]['utf8_string']
        str = str.strip("\n")
        print(str)
        image_text.write(str.encode('utf-8')+' ')
    image_text.close()