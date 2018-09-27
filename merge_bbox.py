import os.path
import glob
import os
import re
import sys
import tarfile
import Image
import time
import yaml
import random

allbbox=[]

with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/models/object_detection/wholeperson_for_pants.txt', 'r') as reader:
    remove = [line.rstrip() for line in reader]

remove=set(remove)

with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/models/object_detection/scopestyle_bbox_all_part1.txt', 'r') as reader:
    box1 = [line.rstrip() for line in reader]
print(len(box1))

with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/models/object_detection/scopestyle_bbox_all_part2.txt', 'r') as reader:
    box2 = [line.rstrip() for line in reader]
    
print(len(box2))

box3=box1+box2
print(len(box3))

for item in box3:
    currentbox = yaml.load(item)
    fileid=currentbox['id']
    if fileid in remove:
        print(fileid)
        box3.remove(item)
        #continue
    #else:
     #   print(fileid)
      #  allbbox.append(item)


print("all box len after remove:", len(box3))



with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/models/object_detection/pants.txt', 'r') as reader:
    pants = [line.rstrip() for line in reader]

print(len(pants))


allbbox=box3+pants

print(len(allbbox))


with open('scopestyle_bbox_all.txt','w') as file:
    for object in allbbox:
        print>>file, object

