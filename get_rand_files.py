import glob
import os
import random
#from random import shuffle

filelist1 = glob.glob("/home/mingming/darknet/VOCdevkit/VOC2019.3/JPEGImages/*")
filelist2 = glob.glob("/home/mingming/darknet/VOCdevkit/VOC2019.2/JPEGImages/*")
filelist3 = glob.glob("/home/mingming/darknet/VOCdevkit/VOC2019.4/JPEGImages/*")

#filelist1.append(filelist2)

filelist = filelist1 + filelist2 + filelist3

random.shuffle(filelist,random.random)

textfile = open('xml_list_2-4.txt','w')
for item in filelist:
    textfile.write("%s\n" % item)
