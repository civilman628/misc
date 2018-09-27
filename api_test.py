from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
import time
import multiprocessing

import numpy as np
import Image
#import matplotlib.pyplot as plt
import requests
from six.moves import urllib
import tensorflow as tf
from StringIO import StringIO
FLAGS = None

class FashionPrediction(object):
    '''class for prediction '''

    def __init__(self):
        with tf.gfile.FastGFile(os.path.join(
                FLAGS.model_dir, 'new_v1.pb'), 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
            self._ = tf.import_graph_def(self.graph_def, name='')

        self.sess1 = tf.Session()
        self.classname = ['men-hoodies',
                          'women-heels',
                          'men-suites',
                          'sneakers',
                          'men-polo',
                          'women-sweaters',
                          'women-trenchcoat',
                          'backpacks',
                          'women-jeans',
                          'men-jeans',
                          'men-sweaters',
                          'women-polo',
                          'men-chinos',
                          'women-maxi',
                          'women-tshirts',
                          'men-messengers',
                          'women-clutch',
                          'men-shirts',
                          'women-blazer',
                          'women-shorts',
                          'women-crossbodybags',
                          'women-wallets',
                          'men-oxfordshoes',
                          'women-skirts',
                          'women-flat',
                          'men-boots',
                          'men-tshirts']
        print('open Server for Fashion Prediction')

    def predict(self, threshold, url, x=-1 , y=-1, w =-1, h =-1):
        '''...'''
        try:
            #response = requests.get(url)
            #img = Image.open(StringIO(response.content))
            #img = Image.open(StringIO(response.content))
	    img = Image.open(FLAGS.image_file)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            if x != -1 and y != -1:
                img = img.crop(x, y, x + w, y + h)
            else:
                resize_image = np.array(img.resize((224, 224), Image.BICUBIC))
                normalize_image = (resize_image - 128.0) / 128.0
                image_4d = np.expand_dims(normalize_image, axis=0)
                #image_array=image_4dTensor.eval(session = sess)

                prediction_tensor = self.sess1.graph.get_tensor_by_name(
                    'final_result:0')  # ADDED

                t1 = time.time()
                predctions = self.sess1.run(
                    prediction_tensor, {'input:0': image_4d})  # ADDED
                predctions = np.squeeze(predctions)  # ADDED
                delta = time.time() - t1

                top_k = predctions.argsort()[-FLAGS.num_top_predictions:][::-1]

                return_data = []


                for class_index in top_k:
                    data = {}
                    #human_string = node_lookup.id_to_string(node_id)
                    score = predctions[class_index]
                    data['score'] = score
                    data['category'] = self.classname[class_index]
                    #print('%s (score = %.5f)' % (human_string, score))
                    if score > threshold:
                        return_data.append(data)
                    print('%s (score = %.5f)' % (self.classname[class_index],score))
                print(delta)

                #return_data['result'] = 'OK'
                #return_data['time'] = delta
                return return_data
        except Exception, e:
            return_data = {}
            return_data['result'] = 'can not make prediction'
            return_data['errorMessage'] = str(e)
            print (e)
            return return_data


class FeatureExtraction(object):
    '''...'''

    def __init__(self):
        with tf.gfile.FastGFile(os.path.join(
                FLAGS.model_dir, 'v1.pb'), 'rb') as f:
            self.graph_def2 = tf.GraphDef()
            self.graph_def2.ParseFromString(f.read())
            self._ = tf.import_graph_def(self.graph_def2, name='')

            self.sess2 = tf.Session()
            print ("open Server for Feature Extraction")

    
    def calculateFeature(self, url, x= -1 , y = -1, w = -1, h = -1):
        '''...'''
        try:
            #response = requests.get(url)
            #img = Image.open(StringIO(response.content))
	    img = Image.open(image_file)	

            if img.mode != 'RGB':
                img = img.convert('RGB')

            if x!= -1 and y!=-1:
                img=img.crop(x, y, x+w, y+h)
            else:
                resize_image = np.array(img.resize((224, 224), Image.BICUBIC))
                normalize_image = (resize_image - 128.0) / 128.0
                image_4d = np.expand_dims(normalize_image, axis=0)
                #image_array=image_4dTensor.eval(session = sess)

                feature_tensor = self.sess2.graph.get_tensor_by_name(
                    'avgpool0/reshape:0')  # ADDED

                t1 = time.time()
                feature_set = self.sess2.run(
                    feature_tensor, {'input:0': image_4d})  # ADDED
                

                feature_set = np.squeeze(feature_set)  # ADDED
                feature_set = feature_set.tolist()

                print(np.size(feature_set))
                print(feature_set)

                #print(delta)
                return_data = {}
                return_data['result'] = 'OK'
                return_data['features'] = feature_set
                return_data['time'] = delta
                return return_data
        except Exception, e:
            return_data = {}
            return_data['result'] = 'ERROR'
            return_data['errorMessage'] = str(e)
            print (e)
            return return_data    

def main(_):
    
    image = (FLAGS.image_file if FLAGS.image_file else
             os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))

    url= 'https://www.tensorflow.org/versions/r0.11/images/grace_hopper.jpg'
    


    #myFeartureExtraction=FeatureExtraction()
    
    #myFeartureExtraction.calculateFeature(url)




    myPrediction=FashionPrediction()

    myPrediction.predict(0.3,url)

    
    # start to load bin fils
    #create_graph()
    # get features
    #run_inference_on_image(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classify_image_graph_def.pb:
    #   Binary representation of the GraphDef protocol buffer.
    # imagenet_synset_to_human_label_map.txt:
    #   Map from synset ID to a human readable string.
    # imagenet_2012_challenge_label_map_proto.pbtxt:
    #   Text representation of a protocol buffer mapping a label to synset ID.
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/v3model/',
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='/home/scopeserver/RaidDisk/fashion_downloader/crawlers/zara_human/men-chinos/6.jpg',
        help='Absolute path to image file.'
    )
    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.'
    )
    FLAGS = parser.parse_args()
    #tf.device("/cpu:0")
    tf.app.run()
