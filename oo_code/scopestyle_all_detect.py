#from video_fashion import fashiondetection
from demo_fashion_7class_new_local import fashiondetection
import numpy as np
import cv2
import Image
import os
import json

#videoformat= cv2.VideoWriter_fourcc(*'MP4V')
#output=cv2.VideoWriter('ouput.mp4',videoformat,25,(1280,720))


myobj = fashiondetection()

#cap = cv2.VideoCapture('video1.mp4')

#fps = cap.get(cv2.CAP_PROP_FPS)

index=0

objectlist=[]


filelist=[]


#for image in sorted(os.listdir("/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/Faster-RCNN_TF/data/VOCdevkit/VOC2007/JPEGImages/")):
#    if image.endswith(".jpg"):
#        image=os.path.join("/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/Faster-RCNN_TF/data/VOCdevkit/VOC2007/JPEGImages/",image)
#        filelist.append(image)

with open('wholeperson_for_pants.txt', 'r') as reader:
    filelist = [line.rstrip() for line in reader]


imagelist=[]


for i in range(0,len(filelist)):
    image=filelist[i]+'.jpg'
    full_path = os.path.join("/home/scopeserver/RaidDisk/DeepLearning/mwang/data/scopestyle_all/", image)
    if os.path.exists(full_path) is False:
        continue
    #img = Image.open(full_path)  # .convert('RGB')
    #if img.mode != 'RGB':
     #   img = img.convert('RGB')
    id=os.path.splitext(os.path.basename(image))[0]
    print('  ')
    print(id)
    
    result = myobj.demo(full_path)
    print(len(result))

    if len(result)>0:
        print('--------')
        for i in result:
            temp={}
            #cv2.rectangle(frame,(i['x'],i['y']),(i['x']+i['w'],i['y']+i['h']),255,2)
            temp['id']=id
            #if i['name']=='leggings':
             #   continue
            temp['name']=i['name']
            temp['x']=i['x']
            temp['y']=i['y']
            temp['w']=i['w']
            temp['h']=i['h']
        #if temp is not None:
            print(temp)
            imagelist.append(temp)

with open('pants.txt','w') as file:
    for object in imagelist:
        print>>file, object



