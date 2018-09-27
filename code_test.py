import os.path, glob, os
import gzip
import pickle
import re
import sys
import tarfile
import Image
import numpy as np
from six.moves import urllib


Minist_URL='http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

CIFAR100_URL='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

dest_directory='./data_file'



def maybe_download_and_extract(DATA_URL):
  """Download and extract model tar file."""
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
  if filepath.find('cifar') > 1: 
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
  else:
    with gzip.open(filepath,mode='rb') as f:
      train,valid,test=pickle.load(f)
      trainx,trainy=train
      #validx,validy=valid
      testx,testy=test
      train_set=np.asarray(trainx,dtype='float32')
      np.save('./data_file/minist_train',train_set)
      print('minist training set is OK ')
      #print(np.max(x))
      test_set=np.asarray(testx,dtype='float32')
      np.save('./data_file/minist_test',test_set)
      print('minist test set is OK ')
      print(train_set.shape)
      print(test_set.shape)


def main():
    for url in [CIFAR100_URL, Minist_URL]:
        maybe_download_and_extract(url)

if __name__ == '__main__':
  main()