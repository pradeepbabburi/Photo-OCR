#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:44:48 2017

@author: babburi
"""

import os
import struct
import numpy as np

def load_mnist(dataset="training", digits=np.arange(10), path=".", size = 60000):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    fimg.close()

#    ind = [ k for k in range(size) if lbl[k] in digits ]
#    N = size #int(len(ind) * size/100.)
#    images = np.zeros((N, rows, cols), dtype=np.uint8)
#    labels = np.zeros((N, 1), dtype=np.uint8)
#    for i in range(N): #int(len(ind) * size/100.)):
#        images[i] = np.array(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols]).reshape((rows, cols))
#        labels[i] = lbl[ind[i]]
#    labels = [label[0] for label in labels]
    return img, lbl