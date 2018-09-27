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

import os.path
import glob
import os
import re
import sys
import tarfile
import Image
import time


os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/v3model/',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'v3.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        




def run_inference_on_image():
    """Runs inference on an image.

    Returns:
      Nothing
    """
    # Creates graph from saved GraphDef.
    #with tf.device('/gpu:3'):
    create_graph()
    #features = []
    #files = []
    #for node in tf.get_default_graph().as_graph_def().node:
     #   print (node.name)

    # filelist=open('/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/tensorflow/models/image/imagenet/searchlist.txt','r')
    with open('women_fake_tops.txt', 'r') as reader:
        filelist = [line.rstrip() for line in reader]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    for image in filelist:
    #for i in range(0, 86495):
        # for image in sorted(os.listdir("/home/scopeserver/RaidDisk/DeepLearning/mwang/data/search_sample/")):
        # image=image.replace("\n","")
        #print (image)

        #image = filelist[i]
        if image.lower().endswith("png"):
            # image=os.path.join("/home/scopeserver/RaidDisk/DeepLearning/mwang/data/search_sample/",image)

            print("\n" + image)

            # if not tf.gfile.Exists(image):
            #tf.logging.fatal('File does not exist %s', image)
            # image_data = tf.gfile.FastGFile(image, 'rb').read().
            # with open(image, 'rb') as f:
            #    image_data = f.read()

            # image_buffer=tf.image.decode_jpeg(image_data)
            # resize_image=tf.image.resize_images(image_buffer,[224,224])
            # unified_image=tf.image.per_image_whitening(resize_image)
            #image_4dTensor=tf.expand_dims(unified_image, 0)
            try:
                # with tf.device('/gpu:3'):  # tf.Session() as sess:
                # with tf.Session(config=config) as sess:  #
                # tf.device("/gpu:3"):
                #sess = tf.Session(config=config)
                img = Image.open(image)  # .convert('RGB')
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                longersize = max(img.size)
                #background = Image.new('RGB', (longersize, longersize), (255,255,255))
                background = Image.new('RGB', (longersize, longersize), "white")
                background.paste(img, (int((longersize-img.size[0])/2), int((longersize-img.size[1])/2)))
                img = background

                #img.save('test.jpg')

                resize_image = np.array(
                    img.resize((299, 299), Image.BICUBIC))
                #resize_image.save('test_299.jpg')    
                normalize_image = (resize_image - 128.0) / 128.0

                image_4d = np.expand_dims(normalize_image, axis=0)
                # stack_vector=np.vstack((image_4d,image_4d,image_4d,image_4d,image_4d))
                # imagearray=image_4dTensor.eval()
                # feature_tensor = sess.graph.get_tensor_by_name('pool_3:0') #ADDED
                # feature_set =
                # sess.run(feature_tensor,{'DecodeJpeg/contents:0':
                # image_data}) #ADDED
                t1 = time.time()
                feature_tensor = sess.graph.get_tensor_by_name(
                    'pool_3:0') # 'avgpool0/reshape:0')  # ADDED
                
                feature_set = sess.run(
                    feature_tensor, {'Mul:0': image_4d})  # ADDED
                delta = time.time() - t1
                feature_set = np.squeeze(feature_set)  # ADDED
                print(np.size(feature_set))
		        #feature_set= np.array(["%.3f" % x for x in feature_set])
                print(feature_set)  # ADDED
                print(delta)
                # files.append(image)
                # features.append(feature_set)
                # os.remove()
                with open("women_fake_tops_list.txt", 'a') as f:
                    # for s in image:
                    f.write(image + '\n')
                with open("women_fake_tops_feature.txt", 'a') as q:
                    # for s in feature_set:
                    #f.write(feature_set +'\n',fmt="%f")
                    np.savetxt(q, feature_set, fmt="%f")
                # sess.close()
                # os.remove(image)
            except Exception, e:
                # sess.close()
                print(e)
                print("error image: " + image)
                # os.remove(image)
                with open("women_fake_tops_error.txt", 'a') as t:
                    t.write(image + '\n')
    # with open("filelist.txt", 'w') as f:
        # f.write("\n".join(files))
    # with open("features.txt", 'w') as q:
        # np.savetxt(q,features,fmt="%f")
    # filelist.close()


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    # maybe_download_and_extract()
    # image = (FLAGS.image_file if FLAGS.image_file else
                    # os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
    # run_inference_on_image(image)
    run_inference_on_image()


if __name__ == '__main__':
    tf.app.run()
