import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
#plt.ioff()
from PIL import Image
import cv2

from collections import namedtuple



rect = namedtuple('rect', 'xmin ymin xmax ymax')


def overlap(a, b):
  dx = min(a['x']+a['w'], b['x']+b['w']) - max(a['x'], b['x'])
  dy = min(a['y']+a['h'], b['y']+b['h']) - max(a['y'], b['y'])
  if (dx>=0) and (dy>=0):
    return dx*dy
  else:
    return 0

def area(a):
  return a['w']*a['h']

def within(a,b):
  a_in_b=False
  b_in_a=False
  if a['x']+a['w']/2.0 >b['x'] and a['x']+a['w']/2.0 < (b['x']+b['w']):
    if a['y']+a['h']/2.0 >b['y'] and a['y']+a['h']/2.0 < (b['y']+b['h']):
      a_in_b=True
  
  if b['x']+b['w']/2.0 >a['x'] and b['x']+b['w']/2.0 < (a['x']+a['w']):
    if b['y']+b['h']/2.0 >a['y'] and b['y']+b['h']/2.0 < (a['y']+a['h']):
      b_in_a=True

  if a_in_b==True and b_in_a==True:
    return True
  else:
    return False

sys.path.append("..")

os.environ["CUDA_VISIBLE_DEVICES"]="2"

#sys.path.append('../models')
#sys.path.append('../models/slim')

from utils import label_map_util
from utils import visualization_utils as vis_util

PATH_TO_CKPT = '/home/scopeserver/RaidDisk/DeepLearning/mwang/models/object_detection/fashion_7classes_new.pb'
test_imagepath='/home/scopeserver/RaidDisk/DeepLearning/mwang/models/object_detection/image/'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/scopeserver/RaidDisk/DeepLearning/mwang/models/object_detection/data/where2buy_7class.pbtxt'
NUM_CLASSES = 7
IMAGE_SIZE = (12, 8)
min_score_thresh=0.7


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for file in sorted(os.listdir(test_imagepath)):
      print(file)
      image = Image.open( os.path.join(test_imagepath,file))
      if image.mode != 'RGB':
        image = image.convert('RGB')
      print image.size
      im_width, im_height = image.size
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = np.asarray(image,dtype='uint8')
      image_np.flags.writeable = True


      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.

      boxes = np.squeeze(boxes)
      classes = np.squeeze(classes).astype(np.int32)
      scores = np.squeeze(scores)
      result=[]
      filter_result=[]
      
      for i in range(boxes.shape[0]):
        if scores is None or scores[i] > min_score_thresh:
          result_dir = {}
          box = tuple(boxes[i].tolist())
          ymin, xmin, ymax, xmax = box
          (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
          result_dir['name'] = str(category_index[classes[i]]['name'])
          result_dir['x'] = int(left)
          result_dir['y'] = int(top)
          result_dir['w'] = int(right-left)
          result_dir['h'] = int(bottom-top)
          result_dir['score'] = scores[i]
          print(result_dir)
          result.append(result_dir)

      new_result=[]

      for item in result:
        if item['name']=='footwear' and item['score'] <0.85 :
          continue
        elif item['name']=='dresses' and item['score'] <0.8 :
          continue
        elif item['name']=='skirt' and item['score'] <0.8 :
          continue
        else:
          new_result.append(item)

      result=new_result
      filter_result=list(result)
      count=len(result)

      if len(result)>1:
        print(count)
        #print('---')
        for i in range(count):
          #print('i=', i)
          item_i=result[i]
          if item_i['name']=='tops' or item_i['name']=='outerwear'or item_i['name']=='dresses' or item_i['name']=='skirts':
            for j in range(i+1,count):
              #print('j=', j)
              item_j=result[j]
              if item_j['name']=='tops' or item_j['name']=='outerwear'or item_j['name']=='dresses' or item_j['name']=='skirts':
                if within(item_i,item_j):
                  if area(item_i)<=area(item_j):
                    if item_i in filter_result:
                      filter_result.remove(item_i)
                      print('remove', item_i['name'])
                  else:
                    if item_j in filter_result:
                      filter_result.remove(item_j)
                      print('remove', item_j['name'])
                elif overlap(item_i,item_j) > min(area(item_i),area(item_j))*0.85 :
                  if area(item_i)<area(item_j):
                    if item_i in filter_result:
                      filter_result.remove(item_i)
                  else:
                    if item_j in filter_result:
                      filter_result.remove(item_j)
          elif item_i['name']=='pants':
            if item_i['h']/item_i['w']<2.0:
              if item_i in filter_result:
               filter_result.remove(item_i)
               print('remove', item_i['name'])
          elif item_i['name']=='footwear' :
             for j in range(i+1,count):
               item_j=result[j]
               if item_j['name']=='footwear':
                  if area(item_i)<=area(item_j):
                    if overlap(item_i,item_j)>area(item_i)*0.9:
                      if item_j in filter_result:
                        filter_result.remove(item_j)
                        print('remove', item_j['name'])
                  else:
                    if overlap(item_i,item_j)>area(item_j)*0.9:
                      if item_i in filter_result:
                        filter_result.remove(item_i)
                        print('remove', item_i['name'])
          elif item_i['name']=='bags' :
             for j in range(i+1,count):
               item_j=result[j]
               if item_j['name']=='bags':
                 if within(item_i,item_j):
                   if area(item_i)<=area(item_j):
                     if item_j in filter_result:
                       filter_result.remove(item_j)
                       print('remove', item_j['name'])
                   else:
                     if item_i in filter_result:                       
                       filter_result.remove(item_i)
                       print('remove', item_i['name'])

      diff=count-len(filter_result)

      print('remove: ',diff)
                


      
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh=min_score_thresh,
          line_thickness=8)
      plt.figure()#figsize=IMAGE_SIZE)
      plt.imshow(image_np) 
      plt.show()
           

