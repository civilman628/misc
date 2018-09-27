# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import Image
import os
import csv
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  
  for node in graph_def.node:
    print (node.name)

  with graph.as_default():
    tf.import_graph_def(graph_def)


  
  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  file_name = "12192013_10392071_1000.jpg"
  model_file = "../image_retraining/filter_model.pb"
  label_file = "../image_retraining/filter_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255.0
  input_layer = "Placeholder"
  output_layer = "final_result"
  #feature = "pool_3"
  #imagelist = "/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/tensorflow/models/image/imagenet/trend_men_imagepart.txt"
  imagelist = './test_list.txt'

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="Placeholder")
  parser.add_argument("--output_layer", help="final_result")
  
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer
  
  graph = load_graph(model_file)
  #sys.exit(0)
  labels = load_labels(label_file)
  '''
  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)
  '''
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)
  '''
  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)
  '''
  sess = tf.Session(graph=graph)

  #print ([n.name for n in tf.get_default_graph().as_graph_def().node])

  label_prob=[]

  with open(imagelist) as f:
    for filename in f:
      img = Image.open(filename.strip())
      if img.mode != 'RGB':
        img = img.convert('RGB')

      longersize = max(img.size)
      background = Image.new('RGB', (longersize, longersize), "white")
      background.paste(img, (int((longersize-img.size[0])/2), int((longersize-img.size[1])/2)))
      img = background
      img = img.resize((input_height, input_width), Image.BICUBIC)
      #img.save('test.jpg')
      img_tesnor = np.array(img)/input_std
      img_tesnor = np.expand_dims(img_tesnor, axis=0)
      
      results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: img_tesnor})
      results = np.squeeze(results)

      top_k = results.argsort()[-1:][::-1]
      index=top_k[0]
      label_prob.append([labels[index],str(results[index])])
      print (labels[index], results[index], filename)
      

  #with open("men_label_prob.csv",'w') as t:
   # fcsv=csv.writer(t)
   # fcsv.writerows(label_prob)



  #for i in top_k:
  #  print(labels[i], results[i])
