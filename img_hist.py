import cv2
import numpy as np



image = cv2.imread("/home/mingming/deep_learning/misc_code/frames_18_left/45Left249N_01991.jpg")

image = cv2.resize(image,(int(image.shape[1]/2),int(image.shape[0]/2)),interpolation=cv2.cv2.INTER_CUBIC)

#cleha = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(128,128))
#image4 = cleha.apply(image)
#cv2.imshow("source",image)
#cv2.waitKey(0)

image1 = image.copy()
for c in range(0,3):
    image1[:,:,c] = cv2.equalizeHist(image[:,:,c])

#image4 = image.copy()
#for c in range(0,3):
 #   image4[:,:,c] = cleha.apply(image[:,:,c])

#cv2.imshow("test", image1)
#cv2.waitKey(0)

image2 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

image2[:,:,2] = cv2.equalizeHist(image2[:,:,2])
image2 = cv2.cvtColor(image2,cv2.COLOR_HSV2BGR)

image3 = np.concatenate((image,image1,image2),axis=1)
cv2.imshow("Source,   RGB Balancing,   HSV Balancing",image3)
#cv2.imshow("test", image2)
cv2.waitKey(0)

