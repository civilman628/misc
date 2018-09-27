import numpy as np
import os
import sys


image_folder="/home/scopeserver/RaidDisk/DeepLearning/mwang/data/scopestyle_all"

with open('scopestyle_all.txt','w') as file:
    for image in sorted(os.listdir(image_folder)):
        if image.endswith(".jpg"):
            #image=os.path.join(image_folder,image)          
            print>>file, image

