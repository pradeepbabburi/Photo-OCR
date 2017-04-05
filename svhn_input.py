# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 19:36:14 2017

@author: babburi
"""

import scipy.io
#from PIL import Image
import numpy as np


mat_train = scipy.io.loadmat('/Users/babburi/Desktop/train/cropped/train_32x32.mat')
mat_test = scipy.io.loadmat('/Users/babburi/Desktop/test/cropped/test_32x32.mat')
X_train = np.array(mat_train['X'])
y_train = np.array(mat_train['y'])
X_test = np.array(mat_test['X'])
y_test = np.array(mat_test['y'])

# replace 10 with 0 in y_train and y_test
y_train[y_train==10] = 0
y_test[y_test==10] = 0

# convert labels into one hot encoding
y_train = np.eye(10)[y_train.astype('int')]
y_train = np.reshape(y_train, [y_train.shape[0], 10])
y_test = np.eye(10)[y_test.astype('int')]
y_test = np.reshape(y_test, [y_test.shape[0], 10])
del mat_train, mat_test

imSize = 32
imPixels = imSize * imSize * 3
n_classes = 10
batch_size = 128

