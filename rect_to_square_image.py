from PIL import Image, ImageOps
#from resizeimage import resizeimage
import os
import numpy as np
import cv2


imagepath='/home/mingming/darknet/VOCdevkit/VOC2019.4/JPEGImages/'



for imagefile in sorted(os.listdir(imagepath)):
    print (imagefile)
    fullname = os.path.join(imagepath,imagefile)
    img = cv2.imread(fullname)

    #background = Image.new('RGB', (768, 768), 'black')
    background = np.zeros((768,768,3),dtype=np.uint8)

    background[:,0:341,:] = img[:,0:341,:]
    background[:,427:768] = img[:,683:1024,:]
    cv2.imwrite(fullname,background)
