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

partial_file_list=[]

file_id_list=[]

for file in filelist:
    index=0
    with open(os.path.join(tagpath,file), 'r') as reader:
        taglist = [line.rstrip() for line in reader]

    for item in taglist:
        index=index+1
        item=item.replace("'", '"')
        print item
        tag=json.loads(item)

        type(tag)
        fileid=tag['id']
        file_id_list.append(fileid)
        classname=tag['name']
        print classname
        x=tag['x']
        y=tag['y']
        w=tag['w']
        h=tag['h']
        imagefile=imagepath+'/'+fileid+'.jpg'
        image = Image.open(imagefile)
        cropimage=image.crop([x,y,x+w,y+h])
        filename=fileid+'.jpg'
        image_part_file=os.path.join(image_part_path,classname,filename)

        if not os.path.isfile(image_part_file):
            cropimage.save(image_part_file, "JPEG", quality=90, optimize=True, progressive=True)
            
        else:
            filename=fileid+'_'+str(index)+'.jpg'
            image_part_file=os.path.join(image_part_path,classname,filename)
            cropimage.save(image_part_file, "JPEG", quality=90, optimize=True, progressive=True)
        
        partial_file_list.append(image_part_file)

        
with open('scopestyle_3orMore_image_part_list.txt','w') as file:
    for object in partial_file_list:
        print>>file, object

source_image_set=set(file_id_list)

with open('scopestyle_3orMore_source_image_list.txt','w') as file2:
    for object in source_image_set:
        print>>file2, object

