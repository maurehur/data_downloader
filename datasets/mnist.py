import os
import urllib.request
import sys
import zipfile
import pandas as pd
import shutil
import gzip
import numpy as np
from scipy.misc import imsave

TRAIN_FEATURES_URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
TRAIN_LABELS_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
TEST_FEATURES_URL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABELS_URL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

#################################################################
# The format is: label, pix-11, pix-12, pix-13, ...
# where pix-ij is the pixel in the ith row and jth column.
#################################################################

def extract_data(filename, num_images):
  """
    Extract the images into a 4D tensor [image index, y, x, channels].
  """
  print('- Extracting', filename)
  real_path = os.path.realpath('data')
  
  with gzip.open(os.path.join(real_path, filename)) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(28 * 28 * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, 28, 28, 1)
    return data

def extract_labels(filename, num_images):
  """
    Extract the labels into a vector of int64 label IDs.
  """
  
  print('- Extracting', filename)

  real_path = os.path.realpath('data')
  with gzip.open(os.path.join(real_path, filename)) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def process(features, labels, mode):

  df = []
  real_path = os.path.realpath('data')
  train_folder = os.path.join(real_path, 'mnist', 'train-images')
  test_folder = os.path.join(real_path, 'mnist', 'test-images')

  if not os.path.exists(train_folder):
    os.makedirs(train_folder)

  if not os.path.exists(test_folder):
    os.makedirs(test_folder)

  if mode == 'test':
    csv_name = mode+'-data'+'.csv'
    print('- Moving Files and built {}'.format(csv_name))
    for i in range(len(features)):
        img_name = str(i) + ".jpg"
        imsave(os.path.join(test_folder, img_name), features[i][:,:,0])
        df.append({'filename': os.path.join(test_folder, img_name), 'class_id': labels[i], 'image':features[i][:,:,0]})
    return pd.DataFrame(df).to_pickle(os.path.join(real_path, 'mnist', csv_name))

  elif mode == 'training':
    csv_name = mode+'-data'+'.csv'
    print('- Moving Files and built {}'.format(csv_name))
    for i in range(len(features)):
        img_name = str(i) + ".jpg"
        imsave(os.path.join(train_folder, img_name), features[i][:,:,0])
        df.append({'filename': os.path.join(train_folder, img_name), 'class_id': labels[i], 'image':features[i][:,:,0]})
    return pd.DataFrame(df).to_pickle(os.path.join(real_path, 'mnist', csv_name))    


def remove_files_mnist(rm):
  dir_name = os.path.realpath('data')
  test = os.listdir(dir_name)
  if rm:
    for item in test:
        if item.endswith(".gz"):
            os.remove(os.path.join(dir_name, item))

    return print('- Removing {}\n'.format('gz files'))
