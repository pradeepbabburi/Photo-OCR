#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:23:37 2017

@author: babburi
"""
import math
import numpy as np

def softICA(W, x, params):
    
    # project weights to norm ball (prevents degenerate bases)
    W_old = W
    W = l2rowScaled(W, 1)
    L1 = np.matmul(W, x)
    L1smooth = np.sqrt(L1**2 + params["epsilon"])
    
    # reconstruction
    L2 = np.matmul(W.T, L1) - x
    cost = (params["lambd"]*np.sum(L1smooth)) + (0.5*np.sum(L2**2))         # ica cost
    grad_1 = np.matmul(np.matmul(W, 2*L2), x.T) + (2*np.matmul(L1, L2.T))   # gradients
    grad_2 = np.matmul(L1/L1smooth, x.T)
    Wgrad = (0.5*grad_1) + (params["lambd"]*grad_2)
    Wgrad = l2rowScaledg(W_old, W, Wgrad, 1)
    return cost, Wgrad

def softICA_lbfgs(W, *args):
    
    x = args[0]
    lambd = args[1]
    epsilon = args[2]
    # project weights to norm ball (prevents degenerate bases)
    W_old = W
    W = l2rowScaled(W, 1)
    L1 = np.matmul(W, x)
    L1smooth = np.sqrt(L1**2 + epsilon)
    
    # reconstruction
    L2 = np.matmul(W.T, L1) - x
    cost = (lambd*np.sum(L1smooth)) + (0.5*np.sum(L2**2))         # ica cost
    grad_1 = np.matmul(np.matmul(W, 2*L2), x.T) + (2*np.matmul(L1, L2.T))   # gradients
    grad_2 = np.matmul(L1/L1smooth, x.T)
    Wgrad = (0.5*grad_1) + (lambd*grad_2)
    Wgrad = l2rowScaledg(W_old, W, Wgrad, 1)
    return cost, Wgrad

def feedForwardRICA(filterDim, poolDim, numFilters, images, W, params):
    """
  feedfowardRICA Returns the convolution of the features given by W with
  the given images. Convolve and pool every image filter pair. 
  There is no bias term b, and the pooling is RICA-style 
  square-square-root pooling instead of average or max pooling.
 
  Parameters:
   filterDim - filter (feature) dimension
   numFilters - number of feature maps
   images - large images to convolve with, tensor in the form
            images(r, c, image number)
   W    - W should be the weights learnt using RICA
          W is of shape (filterDim,filterDim,numFilters)
 
  Returns:
   features - matrix of convolved and pooled features in the form 
   features(imageRow, imageCol, featureNum, imageNum)
     """
    numImages = images.shape[0]
    imageDim = params["imSize"]
    convDim = imageDim - filterDim  + 1
    features = np.zeros(convDim/poolDim, convDim/poolDim, numFilters, numImages)
    poolMat = np.ones(poolDim)
    
    # convolve every image with every filter
    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            _filter = W[:, :, filterNum]
            _image = images[imageNum, :, :]
            _conv = np.convolve(_image, _filter, 'valid')
            act = np.zeros(convDim/poolDim, convDim/poolDim)
            act = np.convolve(_conv**2, poolMat, 'valid')
            act = np.sqrt(act + params["epsilon"])
            act = act[::poolDim, ::poolDim]
            features[:, :, filterNum, imageNum] = act
    return features

def ZCAWhite(images):
    epsilon = 0.1
    sigma = np.dot(images.T, images) / images.shape[0]
    U, s, _ = np.linalg.svd(sigma, full_matrices=True)
    V = np.dot(np.dot(U, np.diag(1/np.sqrt(s + epsilon))), U.T)
    images = np.dot(images, V)
    return images
   
def l2rowScaled(x, alpha):
    eps = 1e-5
    epsSumSq = np.sum(x**2, axis=1, keepdims=True) + eps
    l2rows = np.sqrt(epsSumSq) * alpha
    y = x / l2rows
    return y

def l2rowScaledg(x, y, outderv, alpha):
    eps = 1e-5
    #assert(outderv != [])
    epsSumSq = np.sum(x**2, axis=1, keepdims=True) + eps
    l2rows = np.sqrt(epsSumSq) * alpha
    grad = (outderv/l2rows) - (y * (np.sum(outderv*x, axis=1, keepdims=True)/epsSumSq))
    return grad

def displayImages(images, n_rows, n_cols):
    
    import matplotlib.gridspec as gs
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(n_cols, n_rows))
    grid = gs.GridSpec(nrows=n_rows, ncols=n_cols, wspace=None)
    ax = [fig.add_subplot(grid[i]) for i in range(n_rows*n_cols)]
    grid.update(hspace=0)
    
    for i in range(images.shape[0]):
        ax[i].imshow(images[i,:,:], 'gray')
        ax[i].axis('off')
    
    plt.show()

def random_mini_batches(X, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from input X
    
    Arguments:
    X -- input data, of shape (number of examples, input size)
    mini_batch_size - size of the mini-batches, integer
    seed -- to be consistent with the results
    
    Returns:
    mini_batches of X -- (mini_batch_X)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    k = 0
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    while k <= m:
        start = k
        end = k + mini_batch_size if k + mini_batch_size <= m else m
        mini_batch_X = shuffled_X[start:end, :]
        mini_batches.append(mini_batch_X)
        k += mini_batch_size + 1
    
    
    return mini_batches
    
    