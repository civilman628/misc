import xml.etree.ElementTree as ET
import os

xmlfile_path = '/home/mingming/darknet/VOCdevkit/VOC2019.4/Annotations/'

#utf8_parser = ET.XMLParser(encoding='utf-8')

for xmlfile in sorted(os.listdir(xmlfile_path)):
    print(xmlfile)
    fullname = os.path.join(xmlfile_path,xmlfile)
    tree = ET.parse(fullname)
    root = tree.getroot()

    size = root.find('size')

    size.find('width').text = str(768)


    for obj in root.iter('object'):

        xmlbox = obj.find('bndbox')

        xmin = int(xmlbox.find('xmin').text)

        if xmin > 512:
            xmlbox.find('xmin').text = str(int(xmlbox.find('xmin').text) - 256)
            xmlbox.find('xmax').text = str(int(xmlbox.find('xmax').text) - 256)
    
    tree.write(fullname)
