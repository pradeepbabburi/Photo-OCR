# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import mnist

imSize = 28
patchWidth = 9             # width of a patch
n = patchWidth**2   # dimensionality of input to RICA
lambd = 0.0005         # sparsity cost
numFeatures = 32        # number of filter banks to learn
epsilon = 1e-2  
randsel = np.random.randint(1, 10000, 200)  # a random selecton of samples for visualization

params = {
        "imSize": imSize,
        "patchWidth": patchWidth,
        "n": n,
        "lambd": lambd,
        "numFeatures": numFeatures,
        "epsilon": epsilon
        }

# randomly initialize the parameters
W = np.random.randn(numFeatures, n)*0.01
W = W / np.sqrt(np.sum(W**2, axis=1))

# load mnist database files
images, labels = mnist.load_mnist(dataset="training", path="/Users/babburi/datasets/MNIST")
labels = labels.reshape(-1,1)

numExamples = images.shape[0]

# 50000 images are taken to be the unlabeled set
unlabeledData = images[:50000,:,:]

# rest of 10000 are to be for the labeled set
trainImages = images[50001:,:,:]
trainLabels = labels[50001:,:]

print("size of unlabeled set", unlabeledData.shape)
print("size of training labeled set", trainImages.shape)

#only keep digits 0-4, so that unlabelled dataset has different distribution than the labelled one
