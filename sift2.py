import cv2
import numpy as np
import os.path, glob, os
import time

imagefolder = "/home/scopeserver/RaidDisk/DeepLearning/mwang/LacePattern/"
dict_size = 512
surfPoints_size = 1000

norm = cv2.BFMatcher(cv2.NORM_L2)

#--------------------------------

surf = cv2.xfeatures2d.SURF_create(surfPoints_size)
#surf.extended=True
bow = cv2.BOWKMeansTrainer(dict_size)

for image in sorted(os.listdir(imagefolder)):
  if image.lower().endswith(".jpg"):
     image = os.path.join(imagefolder, image)
     print("\n" + image)
     img = cv2.imread(image)
     [height, width, channels] = img.shape
     #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

     if width>=height:
         x1=(width-height) /2
         x2=x1+height
         y1=0
         y2=height
     else:
         x1=0
         x2=width
         y1=(height-width)/2
         y2=y1+width
          
     img=img[y1:y2, x1:x2]
     img=cv2.resize(img,(800,800),interpolation=cv2.INTER_CUBIC)


     t1 = time.time()
     kp, des = surf.detectAndCompute(img, None)
     #print(len(des))
     
     print(des)
     delta = time.time() - t1
     print(delta)

     if des is not None:
       print (des.shape)
       bow.add(des)

t2 = time.time()
dictionary=bow.cluster()
delta2 = time.time() - t2
print(delta)

np.save("dictionary_surf_512_1000_rgb_s800.npy",dictionary)

''''

bow_descriptor = cv2.BOWImgDescriptorExtractor(surf,norm)
bow_descriptor.setVocabulary(dictionary)

for image in sorted(os.listdir(imagefolder)):
  if image.lower().endswith(".jpg"):
     image = os.path.join(imagefolder, image)
     print("\n" + image)
     img = cv2.imread(image)
     keypoints = surf.detect(img)
     vector =  bow_descriptor.compute(img,keypoints)
     print(vector)

'''


