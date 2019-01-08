#!/usr/bin/env python
# coding: utf-8

# In[14]:


import glob
import codecs
import chardet
from collections import OrderedDict
allTxt=glob.glob(r'synth90k_million/*.jpg')


dic={}
dic["<nul>"]=5000
dicNum={}

j=1

for image_text in allTxt:

        text = image_text.split("_")[2]
        print(text)
        for word in text:
            if word in dic:
                dicNum[word] = dicNum[word]+1
                continue
            dic[word]=j
            dicNum[word] = 1
            j=j+1
print("dict num:"+str(j))
print(dic)

dic = OrderedDict(sorted(dic.items(),key=lambda x:x[1]))
with codecs.open("synth90k_dict.txt",'w',encoding='utf-8') as f_w:
    for key in dic:
        f_w.write(str(dic[key])+"\t"+key+"\n")

with open("synth90k_dict.txt",'rb',encoding='utf-8') as f_r:
    data = f_r.read()
    print(chardet.detect(data))
# dicNum = OrderedDict(sorted(dicNum.items(),key=lambda x:ord(x[0])))
# print(dicNum)
# with open("synth90k_dict_num.txt","w",encoding="utf-8") as f_w:
#     for key in dicNum:
#         f_w.write(key+"\t"+str(dicNum[key])+"\n")




