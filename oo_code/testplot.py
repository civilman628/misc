import matplotlib.pyplot as plt
#plt.rcParams['backend']="Qt4Agg"
plt.get_backend()
#plt.switch_backend("TkAgg")
from PIL import Image
img = Image.open('/home/scopeserver/RaidDisk/DeepLearning/mwang/models/object_detection/test_image/81806.jpg')
imgplot = plt.imshow(img)
plt.show()
