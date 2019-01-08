#!/usr/bin/env python3
# coding:utf-8

import json
import os

annotationFile = "COCO_Text.json"
#annotationFile = "cocotext.v2.json";

data = json.dumps(annotationFile, sort_keys=True, indent=4, separators=(',',':'))
print(type(data))
print(data[0:1000])

annotations = json.load(open(annotationFile, 'r'))
data1 = json.dumps(annotationFile, sort_keys=True, indent=4, separators=(',',':'))

print(type(data1))

print(data1[0:1000])
textAnnot = {int(annid):annotations['anns'][annid] for annid in annotations['anns']}
# print(type(textAnnot),textAnnot)
print(type(textAnnot))


flag=0
for key,value in textAnnot.items():
	if flag:
		break
	print(key,value)
	flag=flag+1

imgToAnnot = {int(cocoid):annotations['imgToAnns'][cocoid] for cocoid in annotations['imgToAnns']}
#print(type(imgToAnnot),imgToAnnot)
print(type(imgToAnnot))

flag=0
for key,value in imgToAnnot.items():
	if flag:
		break
	print(key,value)
	flag=flag+1

imgs = {int(cocoid):annotations['imgs'][cocoid] for cocoid in annotations['imgs']}
#print(type(imgs),imgs)
print(type(imgs))
flag=0
for key,value in imgs.items():
	if flag:
		break
	print(key,value)
	flag=flag+1
res = {}


'''
res = {
    'file1.jpg':{
        'box_num':2,
        'annotations':[
            {
                "utf8_string":"string in box1",
                "language":"english",
                "bbox":[left_bottom_x, left_bottom_y, width, height],
                "rotate_clockwise":0,
                "polygon":[x0, y0, x1, y1, x2, y2, ...],
                },
            {
                "utf8_string":"string in box2",
                "language":"english",
                "bbox":[left, top, width, height],
                "rotate_clockwise":0,
                "polygon":[x0, y0, x1, y1, x2, y2, ...],
                },
            ],
        'width':224,
        'height':224,
        'other_info':'...',
        ...
        },
    'file2.jpg':{
        'box_num':0,
        'annotations':[
            {
                "utf8_string":"This is a string",
                "language":"english",
                }
            ],
        'width':224,
        'height':224,
        'other_info':'...',
        },

    'file3.jpg':{
        ...
    },
    ...
    }

'''
'''
for cocoid in imgToAnnot:
    annids = imgToAnnot[cocoid]
    try:
        anns = [textAnnot[annid] for annid in annids]
        #print(cocoid)
        filename = imgs[cocoid]['file_name']
        res[filename] = {'annotations':[]}
        for ann in anns:
            if('utf8_string' in ann):
                #print(ann)
                annotation = {
                    'utf8_string':ann['utf8_string'],
                    'language':ann['language'],
                    'bbox':ann['bbox'], # [left, top, width, height]
                    'rotate_clockwise':0,
                    'polygon':ann['polygon'], # [x0, y0, x1, y1, x2, y2, ...]
                    'area':ann['area'],
                    'legibility':ann['legibility'],
                    'class':ann['class']
                }
                res[filename]['annotations'].append(annotation)
        if not res[filename]['annotations']:
            del res[filename]
        else:
            res[filename]['width'] = imgs[cocoid]['width']
            res[filename]['height'] = imgs[cocoid]['height']
            #imgSet = imgs[cocoid]['set']
#            print("{}\t{}".format(imgSet, filename))
            #os.system('cp /data130/ocr/coco-text/*/{} /data130/ocr/coco-text/trainV2'.format( \
            #                                                                                       filename))
    except:
        #print(cocoid)
        pass

with open('coco_annot.json', 'w') as f:
    json.dump(res, f)
'''