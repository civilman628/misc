
import os.path, glob, os
#import sys
#import tarfile
import Image
import json
#import string

imagepath = '/home/scopeserver/RaidDisk/DeepLearning/mwang/data/scopestyle_all'

tagpath = '/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/tensorflow/models/image/imagenet'

image_part_path = '/home/scopeserver/RaidDisk/DeepLearning/mwang/data/scopestyle_8_classes_images'

filelist = ['scope_3and_more_1.txt','scope_3and_more_2.txt','scope_3and_more_3.txt']

classname_list=[]

for file in filelist:
    index=0
    with open(os.path.join(tagpath,file), 'r') as reader:
        taglist = [line.rstrip() for line in reader]

    for item in taglist:
        item=item.replace("'", '"')
        tag=json.loads(item)
        classname=tag['name']
        classname_list.append(classname)


with open('scopestyle_3orMore_classname_list.txt','w') as file:
    for object in classname_list:
        print>>file, object