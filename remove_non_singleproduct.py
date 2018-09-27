
import glob
import os
from PIL import Image
import numpy as np



directory = "/home/scopeserver/RaidDisk/Filter_Full/"
file_paths = []  # List which will store all of the full filepaths.

# Walk the tree.
#textfile = open('single_product.txt','w')

for root, directories, files in os.walk(directory):
    for filename in files:
        # Join the two strings in order to form the full filepath.
        filepath = os.path.join(root, filename)
        #file_paths.append(filepath)  # Add it to the list.
        #textfile.write("%s\n" % filepath)
        print(filepath)
        img = np.array(Image.open(filepath))
        img = np.mean(img,2)
        if np.abs(np.max(img[0,:])-np.min(img[0,:]))>30:
            os.remove(filepath)
            continue
        elif np.abs(np.max(img[-1,:])-np.min(img[-1,:]))>30:
            os.remove(filepath)
            continue
        elif np.abs(np.max(img[:,0])-np.min(img[:,0]))>30:
            os.remove(filepath)
            continue
        elif np.abs(np.max(img[:,-1])-np.min(img[:,-1]))>30:
            os.remove(filepath)
            continue