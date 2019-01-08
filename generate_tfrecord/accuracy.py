#!/usr/bin/env python3
# coding:utf-8

import os
import sys
import json
import numpy as np
from Levenshtein import *

recogFile = sys.argv[1]
annotFile = sys.argv[2]
#recogFile = "coco_test.txt"
#annotFile = "/data/coco-text/coco_annot.json"

annot = json.load(open(annotFile, 'r'))
totalLD = 0.0
totalLen = 0.0
accs = []

with open(recogFile, 'r') as fin:
    for line in fin:
        recogText = ""
        try:
            imgFile, recogText = line.strip().split("\t")
        except:
            imgFile = line.strip()
        imgFile = os.path.basename(imgFile)
        imgAnnots = annot[imgFile]['annotations']
        annotString = ''
        for imgAnnot in imgAnnots:
            annotString += imgAnnot['utf8_string'].lower()
            totalLen += (len(imgAnnot['utf8_string']))
        ld = distance(recogText, annotString)
        annotLen = len(annotString)
        acc = float(annotLen - ld) / annotLen
        totalLD += ld
        accs.append(acc)

aveAccRate = np.mean(accs)
totalAccRate = (totalLen - totalLD) / totalLen
print("file:{}".format(recogFile))
print("average accuracy:{}".format(aveAccRate))
print("total accuracy:{}".format(totalAccRate))
