from PIL import Image, ImageOps
from resizeimage import resizeimage
import os


imagepath='/home/scopeserver/RaidDisk/DeepLearning/mwang/progressive_growing_of_gans/dresses/'

#longersize=256
'''
for imagefile in sorted(os.listdir(imagepath)):
    print imagefile
    fullname = os.path.join(imagepath,imagefile)
    img = Image.open(fullname)
    longersize=max(img.size)
    
    background = Image.new('RGB', (longersize, longersize), 'white')
    background.paste(img, (int((longersize-img.size[0])/2), int((longersize-img.size[1])/2)))
    img = background  
    img.save(fullname,'JPEG')
'''


'''
with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/tensorflow/models/image/imagenet/107_classes_image_list.txt','r') as reader:
    filelist = [line.rstrip() for line in reader]

for fullname in filelist:
    print fullname
    #fullname = os.path.join(imagepath,imagefile)
    img = Image.open(fullname)
    longersize=max(img.size)
    
    background = Image.new('RGB', (longersize, longersize), 'white')
    background.paste(img, (int((longersize-img.size[0])/2), int((longersize-img.size[1])/2)))
    img = background  
    img.save(fullname,'JPEG', )
'''


for imagefile in sorted(os.listdir(imagepath)):
    print imagefile
    fullname = os.path.join(imagepath,imagefile)
    img = Image.open(fullname)

    longersize=max(img.size)
    
    background = Image.new('RGB', (longersize, longersize), 'white')
    background.paste(img, (int((longersize-img.size[0])/2), int((longersize-img.size[1])/2)))
    img = background

    img = img.resize((256,256),Image.BICUBIC)
    #longersize=max(img.size)
    
    #background = Image.new('RGB', (longersize, longersize), 'white')
    #background.paste(img, (int((longersize-img.size[0])/2), int((longersize-img.size[1])/2)))
    #img = background
    img.save(fullname,'JPEG')
