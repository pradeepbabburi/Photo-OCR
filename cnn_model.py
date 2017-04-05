# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 19:28:26 2017

@author: babburi
"""

import tensorflow as tf
import svhn_input

# variables
n_classes = svhn_input.n_classes
imSize = svhn_input.imSize
imPixels = svhn_input.imPixels
batch_size = svhn_input.batch_size

# tf graph input
x = tf.placeholder(tf.float32, [imSize, imSize, 3, None])
y = tf.placeholder(tf.float32, [None, n_classes])

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# pooling
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# build cnn model
def convolutional_neural_network(x):
    weights = {
        # 5x5 convolution, 3 channels, 64 outputs
        'W_conv1': tf.Variable(tf.truncated_normal([5,5,3,32], stddev=0.05)),
        
        # 5x5 convolution, 64 inputs, 64 outputs
        'W_conv2': tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.05)),
        
        # fully connected layer, 8x8x64 inputs, 384 outputs
        'W_fc3': tf.Variable(tf.truncated_normal([8*8*64,512], stddev=0.04)),
        
        # fully connected layer, 384 inputs, 192 outputs
        'W_fc4': tf.Variable(tf.truncated_normal([512,256], stddev=0.04)),
                                                 
        # 192 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.truncated_normal([256, n_classes], stddev=1/256.0))
    }
    
    biases = {
        'b_conv1': tf.Variable(tf.constant(0.0, shape=[32])),
        'b_conv2': tf.Variable(tf.constant(0.0, shape=[64])),
        'b_fc3': tf.Variable(tf.constant(0.1, shape=[512])),
        'b_fc4': tf.Variable(tf.constant(0.1, shape=[256])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
    }
    
    # reshape input to a 4D tensor
    x = tf.reshape(x, shape=[-1, imSize, imSize, 3])
    
    # convolution layer 1 followed by pooling
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    pool1 = maxpool2d(conv1)
    #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    
    # convolution layer 2 followed by pooling
    conv2 = tf.nn.relu(conv2d(pool1, weights['W_conv2']) + biases['b_conv2'])
    #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    pool2 = maxpool2d(conv2)
    
    # fully connected layer 3
    fc3 = tf.reshape(pool2, shape=[-1, 8*8*64])
    fc3 = tf.nn.relu(tf.matmul(fc3, weights['W_fc3']) + biases['b_fc3'])
    
    # fully connected layer 4
    fc4 = tf.nn.relu(tf.matmul(fc3, weights['W_fc4']) + biases['b_fc4'])
    
    # output layer
    output = tf.matmul(fc4, weights['out']) + biases['out']
    return output