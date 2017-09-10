#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:38:58 2017

@author: babburi
"""

import numpy as np
import mnist
import stl_utils

# load mnist data
X, y = mnist.load_mnist(dataset="testing", path="/Users/babburi/datasets/MNIST")

# initialize hyperparameters
imSize = 28
n = imSize**2
m = X.shape[0]
lambd = 0.0005
numFeatures = 50
epsilon = 1e-2
randsel = np.random.randint(1, m, 50)  # a random selecton of samples for visualization

params = {
        "n": n,
        "m": m,
        "imSize": imSize,
        "lambd": lambd,
        "numFeatures": numFeatures,
        "epsilon": epsilon
        }

# display raw images
print("sampled raw images")
stl_utils.displayImages(X[randsel,:,:], 5, 10)

# preprocess images - zca whitened
zcaX = stl_utils.ZCAWhite(X.reshape(-1, n))
zcaX = zcaX.reshape(-1,imSize,imSize)
print("sampled images after zca whitening")
stl_utils.displayImages(zcaX[randsel,:,:], 5, 10)

# normalize images
_norm = np.sqrt(np.sum(zcaX**2) + (1e-8))
zcaX= zcaX/_norm
zcaX = zcaX.reshape(-1,n)

# randomly initialize the weights
W = np.random.randn(numFeatures, n)*0.01
W = W / np.sqrt(np.sum(W**2, axis=1, keepdims=True))

# optimize weights using gradient descent
num_iter = 300
learning_rate = 40.0
n_epochs = 300
for epoch in range(n_epochs+1):
    epoch_loss = 0
    mini_batches = stl_utils.random_mini_batches(zcaX)
    for mini_batch in mini_batches:
        mini_batch_X = mini_batch
        cost, grad = stl_utils.softICA(W, mini_batch_X.T, params)
        W = W - (learning_rate * grad)
        epoch_loss += cost
    
    if epoch % 100 == 0:
        print("Cost after epoch %i: %f" %(epoch, epoch_loss))
        # visualize weights
        print("filters after epoch %i:" %(epoch))
        stl_utils.displayImages(W.reshape(-1, imSize, imSize), 5, 10)
        learning_rate /= 2





