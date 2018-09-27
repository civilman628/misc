import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

import urllib
from StringIO import StringIO
import requests
import Image

os.environ["CUDA_VISIBLE_DEVICES"]="2"

sys.path.append('/home/scopeserver/RaidDisk/DeepLearning/mwang/models')
sys.path.append('/home/scopeserver/RaidDisk/DeepLearning/mwang/models/slim')

from collections import defaultdict
#from matplotlib import pyplot as plt

from utils import label_map_util
from utils import visualization_utils as vis_util

PATH_TO_CKPT='/home/scopeserver/RaidDisk/DeepLearning/mwang/models/object_detection/output_inference_graph.pb'
PATH_TO_LABELS='/home/scopeserver/RaidDisk/DeepLearning/mwang/models/object_detection/data/scope_label_map.pbtxt'
NUM_CLASSES = 18
min_score_thresh = 0.5


class fashiondetection(object):
    
    def __init__ (self):

            
        # init session
        #self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # load network
        #self.net = get_network(args.demo_net)
        # load model
        #saver = tf.train.Saver()
        #saver.restore(self.sess, args.model)
        #sess.run(tf.initialize_all_variables())

        #print '\n\nLoaded network {:s}'.format(args.model)

        # Warmup on a dummy image
        #im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
        #for i in xrange(2):
         #   _, _= im_detect(self.sess, self.net, im)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        #im_names = ['70.jpg','52.jpg','4.jpg','demo.jpg','7.jpg','8.jpg','9.jpg','13.jpg']


        #for im_name in im_names:
        #   print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        #  print 'Demo for data/demo/{}'.format(im_name)
        # demo(sess, net, im_name)

        #plt.show()

    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    

    def demo(self, url):
        """Detect object classes in an image using pre-computed object proposals."""
        print url
        result=[]
        #response = urllib.urlopen(url)
	    #timer1 = Timer()
	    #timer1.tic()
		full_path = os.path.join("/home/scopeserver/RaidDisk/SocialMediaImages", url)
        image=Image.open(full_path)
        im_width, im_height = image.size
        print(im_height,im_width)
        image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        t1 = time.time()

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                t1 = time.time()
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})  
                delta=time.time()-t1
                print(delta)

                boxes = np.squeeze(boxes)
                classes = np.squeeze(classes).astype(np.int32)
                scores = np.squeeze(scores)
                for i in range(boxes.shape[0]):
                    if scores is None or scores[i] > min_score_thresh:
                        result_dir = {}
                        box = tuple(boxes[i].tolist())
                        ymin, xmin, ymax, xmax = box
                            
                        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                            ymin * im_height, ymax * im_height)
                        if self.category_index[classes[i]]['name']=='feet' or self.category_index[classes[i]]['name']=='leg' :
                            continue
                        else:
                            result_dir['name'] = self.category_index[classes[i]]['name']
                            result_dir['x'] = int(left)
                            result_dir['y'] = int(top)
                            result_dir['w'] = int(right-left)
                            result_dir['h'] = int(bottom-top)
                            result.append(result_dir)   
                                

                            
         

                    


        
        return result


        # Detect all object classes and regress object bounds
        '''
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.sess, self.net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
            '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class
        im = im[:, :, (2, 1, 0)]
        #fig, ax = plt.subplots(figsize=(12, 12))
        #plt.show()
        #ax.imshow(im, aspect='equal')
        CONF_THRESH = 0.60
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                            cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            #vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) != 0:
                for i in inds:
                    bbox = dets[i, :4]
                    #bbox = bbox.tolist()
                    x = bbox[0]
                    y = bbox[1]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    result_dir = {}
                    #
                    #cropping = im[bbox[0]:bbox[2],bbox[1],bbox[3]]
                    #result_dir['image']=base64.b64encode(cropping)
                    #
                    result_dir['name'] = [CLASSES[cls_ind]]
                    result_dir['x'] = int(x)
                    result_dir['y'] = int(y)
                    result_dir['w'] = int(w)
                    result_dir['h'] = int(h)


                    #score = dets[i, -1]
                    #np.concatenate((result,[CLASSES[cls_ind],bbox]),axis=0)
                    result.append(result_dir)
                    #np.vstack((result,[cls_ind,bbox]))
                    #np.concatenate((result,[cls_ind,bbox]))
                    '''
        return result

#============================================

    def demobase64(self, imagearray):
        """Detect object classes in an image using pre-computed object proposals."""

        result=[]
        
        first_coma=imagearray.find(',')
        img_bytes=base64.decodestring(imagearray[first_coma:])
        im=np.asarray(bytearray(img_bytes),dtype='uint8')
        im=cv2.imdecode(im,cv2.IMREAD_COLOR)
        #response = urllib.urlopen(url)
	    #timer1 = Timer()
	    #timer1.tic()
        #response = requests.get(url)
        #im=Image.open(StringIO(response.content))
	    #timer1.toc()
	    #print("get image from url, time cost: ", timer1.total_time)
        #im=np.asarray(im)
        #im=np.asarray(bytearray(response.read()),dtype="uint8")
        #image = cv2.imdecode(im,cv2.IMREAD_COLOR)
        # Load the demo image
        #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
        #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
        #im = cv2.imread(im_file)

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.sess, self.net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
            '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class
        im = im[:, :, (2, 1, 0)]
        #fig, ax = plt.subplots(figsize=(12, 12))
        #plt.show()
        #ax.imshow(im, aspect='equal')
        CONF_THRESH = 0.60
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                            cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            #vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) != 0:
                for i in inds:
                    bbox = dets[i, :4]
                    #bbox = bbox.tolist()
                    x = bbox[0]
                    y = bbox[1]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    result_dir = {}
                    result_dir['name'] = [CLASSES[cls_ind]]
                    result_dir['x'] = int(x)
                    result_dir['y'] = int(y)
                    result_dir['w'] = int(w)
                    result_dir['h'] = int(h)
                    #score = dets[i, -1]
                    #np.concatenate((result,[CLASSES[cls_ind],bbox]),axis=0)
                    result.append(result_dir)
                    #np.vstack((result,[cls_ind,bbox]))
                    #np.concatenate((result,[cls_ind,bbox]))
        return result
        


#def main():
 #   myobj=fashiondetection()
 #   myobj.demo('https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAgPAAAAJDM4M2ZkODU3LTc1YzYtNDM3Yi04MmE3LWUyYzQzN2UxMDc0OA.jpg')

#if __name__ == "__main__":
 #   main()
