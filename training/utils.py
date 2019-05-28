### IMPORTS
# standard imports
from __future__ import division,print_function
import os, json, sys
from glob import glob
import pdb
from time import time, sleep, strftime
from random import shuffle
from distutils.dir_util import copy_tree
import copy
import shutil
from shutil import rmtree, copytree, move, copyfile
import re
import bcolz
import itertools

# data science imports: numpy, pandas, scipy, sklearn
import numpy as np
from numpy.random import random, permutation
import pandas as pd
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import PIL
from PIL import ImageEnhance, Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score

# import for display
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
from IPython.display import clear_output
from IPython import get_ipython
from tqdm import *
#import mpld3
#mpld3.enable_notebook()
from IPython.display import Image as ImageJup
plt.rcParams['figure.figsize'] = [18,3]

# keras imports
import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model, load_model
from keras.layers import Input, BatchNormalization, concatenate, Conv2D, TimeDistributed, LSTM, Conv3D, MaxPooling3D, ZeroPadding3D, CuDNNLSTM, Cropping1D, Conv1D, Activation, multiply, CuDNNGRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam, Nadam
from keras.losses import mean_absolute_percentage_error, mean_squared_error
from keras.preprocessing.image import img_to_array
from keras.callbacks import TensorBoard
from keras.applications import xception, inception_resnet_v2, vgg16
import tensorflow as tf
from keras import activations
import matplotlib.cm as cm

# set the seed
np.random.seed(42)

### FUNCTIONS
def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)

def plot(img):
    plt.imshow(to_plot(img))


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def wrap_config(layer):
    return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def save_mult_arrays(dir_name, arr_tuple):
    """same as save_array only with tuple of arrays """
    if not os.path.exists(dir_name): os.mkdir(dir_name)
    i = 0
    for arr in arr_tuple:
        fname = dir_name + '/tup{}.bc'.format(i)
        c=bcolz.carray(arr, rootdir=fname, mode='w')
        c.flush()
        i = i +1

def load_mult_arrays(dir_name, verbose=True):
    """same as load_array only with tuple of arrays."""
    l = []
    arr_names = sorted(glob(dir_name + '/*'))
    if verbose: print('Found {} arrays'.format(len(arr_names)))
    for arr in arr_names:
        f = bcolz.open(arr)[:]
        if isinstance(f[0],dict) and len(f)==1: f = f[0]
        l.append(f)
    return l

def clear_vars(var_list=[]):
    for var in var_list:
        try: del globals()[var]
        except KeyError: print('Variable not in globals')

def names_of_biggest_variables(no_vars = 5):
    """ get the names of the local biggest variables"""
    variables, sizes = [], []
    for var, obj in globals().items():
        variables.append(var)
        sizes.append(sys.getsizeof(obj))
    # get the sorted idcs
    idcs = np.argsort(sizes)
    # print the names
    print('Names: ' + str([variables[i] for i in idcs][::-1][:no_vars]))
    print('Sizes: ' + str([sizes[i] for i in idcs][::-1][:no_vars]))

def double_dict_to_list(d):
    n = len(d['red_light'])
    new_d = [{k:d[k][i]for k in d.keys()} for i in range(n)]
    return new_d

def save_unpacked_mult_arrays(dir_name, name_arr, data):
    """same as save_array only with tuple of arrays """
    if not os.path.exists(dir_name): os.mkdir(dir_name)
    for n, d in zip(name_arr, data):
        im_name = re.findall('image_\d+',n)[0]
        fname = dir_name + '/{}.bc'.format(im_name)
        if isinstance(d, tuple):
            save_mult_arrays(fname, d)
        else:
            c=bcolz.carray(d, rootdir=fname, mode='w')
            c.flush()

def convert_input(im, for_prediction=True):
    """
    input = PIL instance
    return resized and reshaped input
    """
    # convert to PIL instance and resize/crop
    im = im.crop((0,120,800,480)).resize((222,100))

    if for_prediction:
        x = np.asarray(im, dtype=K.floatx())
        x = x.transpose(2, 0, 1)
        x = np.expand_dims(x,0)
    else:
        x= np.asarray(im)

    return x

