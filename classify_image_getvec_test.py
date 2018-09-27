# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
import time

import numpy as np
import Image
#import matplotlib.pyplot as plt
import requests
from six.moves import urllib
import tensorflow as tf
from StringIO import StringIO
FLAGS = None


DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'v1.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
    """Runs inference on an image.

    Args:
      image: Image file name.

    Returns:
      Nothing
    """
    # if not tf.gfile.Exists(image): https://www.tensorflow.org/versions/r0.11/images/grace_hopper.jpg
    #tf.logging.fatal('File does not exist %s', image)
    #image_data = tf.gfile.FastGFile(image, 'rb').read()

    # with open(image, 'rb') as f:
    #image_data = f.read()

    # /home/scopeserver/RaidDisk/downloader/node_crawler/download/1478129340373.jpg

    #-------------------url--------------------
    #response = requests.get('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/800px-President_Barack_Obama.jpg')
    img = Image.open(image)  # .convert('RGB')

    if img.mode != 'RGB':
        img = img.convert('RGB')

    resize_image = np.array(img.resize((224, 224), Image.BICUBIC))
    normalize_image = (resize_image - 128.0) / 128.0
    #print (normalize_image.min())
    #print (min(img))
    # normalize_image=Image.fromarray(normalize(img))
    image_4d = np.expand_dims(normalize_image, axis=0)
    # copy_4d=image_4d
    # for i in range (1,128):
    # copy_4d=np.concatenate((copy_4d,image_4d),axis=0)
    #-------------------end url----------------

    # image_buffer=tf.image.decode_jpeg(response.content)
    # resize_image=tf.image.resize_images(image_buffer,[224,224],method=2)

    # plt.imshow(tf.image.encode_jpeg(resize_image))
    # unified_image=tf.image.per_image_whitening(resize_image)
    #image_4dTensor=tf.expand_dims(resize_image, 0)

    #x =np.ones([224,224,3],dtype=int);
    #y = np.expand_dims(x, axis=0)

    # Creates graph from saved GraphDef.
    # load bin files
    # x=tf.Variable(1.0)
    #saver = tf.train.Saver()
    # tf.device('/gpu:3')

    # for tensor in tf.get_default_graph().as_graph_def().node:
    # print(tensor.name)

    with tf.device('/gpu:3'):  # tf.Session() as sess:
        #saver.restore(sess, "/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/v2model/inception_v2.ckpt")
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the
        # graph.

        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #sess = tf.Session(config=config)  # as sess
        sess = tf.Session()
        #-----------------------------------------
        # stamp1=time.time()
        # image_array=image_4dTensor.eval()
        # time_cost=time.time()-stamp1
        #print (time_cost)

        # t1=time.time()

        #softmax_tensor = sess.graph.get_tensor_by_name('softmax2:0')
        # predictions = sess.run(softmax_tensor,
        #{'input:0': img2})
        #predictions = np.squeeze(predictions)
        # delta=time.time()-t1
        # print(delta)

        #------------------------------------------
        # image_array=image_4dTensor.eval()
        

        feature_tensor = sess.graph.get_tensor_by_name(
            'final_result:0')  # 'avgpool0/reshape:0')  # ADDED
        feature_tensor2 = sess.graph.get_tensor_by_name('avgpool0/reshape:0')
        
        t1 = time.time()
        feature_set = sess.run(feature_tensor, {'input:0': image_4d})  # ADDED
        delta = time.time() - t1
        feature_set = np.squeeze(feature_set)  # ADDED
        #feature_set2 = np.squeeze(feature_set2) 
        #print(np.size(feature_set2))
        # print(np.size(feature_set2))
        #feature_set = feature_set.tolist()
        #print(feature_set2)
        
        print(delta)

        top_k = feature_set.argsort()[-FLAGS.num_top_predictions:][::-1]

        for node_id in top_k:
            #human_string = node_lookup.id_to_string(node_id)
            score = feature_set[node_id]
            #print('%s (score = %.5f)' % (human_string, score))
            print(' (score = %.5f)' % score)


def main(_):
    # maybe_download_and_extract()
    image = (FLAGS.image_file if FLAGS.image_file else
             os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))

    # start to load bin fils
    create_graph()
    # get features
    run_inference_on_image(image)


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
        default='',
        help='Absolute path to image file.'
    )
    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.'
    )
    FLAGS = parser.parse_args()
    tf.device("/cpu:0")
    tf.app.run()
