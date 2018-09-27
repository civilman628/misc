from flask import Flask
from flask import request 
from flask import jsonify
import tensorflow as tf 
import os
import argparse
import requests
import numpy as np
import Image
from StringIO import StringIO
import time
FLAGS = None
app = Flask(__name__)



def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'v1.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

@app.route('/')
def index():
    return "Hello World!"

@app.route('/api/getfeature', methods=['POST'])
def create_task():
    if not request.json or not 'url' in request.json:
        abort(400)
    task = calculateFeature(request.json['url'])
    return jsonify(task), 201

def calculateFeature(url):
    try:
        response = requests.get(url)    
        img=Image.open(StringIO(response.content))
        resize_image=np.array(img.resize((224,224),Image.BICUBIC))
        normalize_image=(resize_image-128.0)/128.0
        image_4d=np.expand_dims(normalize_image, axis=0)
        #image_array=image_4dTensor.eval(session = sess)
        t1=time.time()
        feature_tensor = sess.graph.get_tensor_by_name('avgpool0/reshape:0') #ADDED
        feature_set = sess.run(feature_tensor,{'input:0': image_4d}) #ADDED
        feature_set = np.squeeze(feature_set) #ADDED
        feature_set = feature_set.tolist()
        #print type(feature_set)
        #print(np.size(feature_set))
        #print(feature_set)
        delta=time.time()-t1
        #print(delta)
        return_data = {}
        return_data['data'] = feature_set
        return_data['time'] = delta
        return return_data
    except:
        return_data['data'] = 'Load Image Error'
        return return_data

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
    #create_graph()
    sess = tf.Session()
    app.run(debug=True)
