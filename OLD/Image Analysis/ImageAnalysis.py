# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:21:17 2022

@author: Raphael
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, restoration, filters, segmentation, feature, exposure, img_as_float
from scipy import ndimage as ndi
from scipy.signal import argrelextrema
import os


# read image into np.array
dataloc = "E:/Raphael/OpenData"
gt = os.path.join(dataloc,"groundtruth")
raw = os.path.join(dataloc,"rawimages")

for img in os.listdir(raw):
    if "normal" in img:
        imID = int(img.split("_")[1])
        
caller = getattr(data, 'human_mitosis')
image = caller()
#plt.figure()
#plt.imshow(image, cmap=plt.cm.gray)


# Denoising

def denoising(image):
    #rolling ball background subtraction 
    
    background = restoration.rolling_ball(image, radius = 50)
    #plt.figure()
    #plt.imshow(background)
    nimage = image - background
    nimage = exposure.equalize_adapthist(nimage)
    return nimage

image = img_as_float(denoising(image))

# Traditional threshholding + watershed

def watershedding(binary):
    distance = ndi.distance_transform_edt(binary)
    #plt.figure()
    #plt.imshow(distance, cmap=plt.cm.gray)
    coords = feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary)
    print(coords)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    #plt.figure()
    #plt.imshow(mask, cmap=plt.cm.gray)
    markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-distance, markers, mask=binary)
    return labels

def threshholding(nimage):
    
    
    #plt.figure()
    #plt.imshow(nimage, cmap=plt.cm.gray)
    
    # Thresholding
    
    thresholds = {}
    thresholds["isodata"] = filters.threshold_isodata(nimage)
    thresholds["li"] = filters.threshold_li(nimage)
    #thresholds["local"] = filters.threshold_local(nimage, block_size=51)
    thresholds["mean"] = filters.threshold_mean(nimage)
    thresholds["min"] = filters.threshold_minimum(nimage)
    thresholds["otsu"] = filters.threshold_otsu(nimage)
    label = {}
    for key in thresholds:
        binary = nimage > thresholds[key]
        #plt.figure()
        #plt.imshow(binary, cmap=plt.cm.gray)
        #plt.title(key)

    
        # Watershed
        
        labels = watershedding(binary)
        #plt.figure()
        #plt.imshow(labels, cmap=plt.cm.nipy_spectral)
        label[key] = labels
    return label
    
lable_thresh = threshholding(image)
# Markers + Watershed

def gethist(image):
    # Get local maxima of histogram
    # defines lower and upper bound for marker selection
    hist = exposure.histogram(image) 
    #plt.figure()
    #plt.plot(hist[1], hist[0])
    x = hist[0]
    arrlen = np.sum(x)
    sort = []
    for i in range(len(x)):
        for j in range(x[i]):
            sort.append(hist[1][i])
    lb = sort[round(arrlen*0.05)]
    ub = sort[round(arrlen*0.95)]
    return lb, ub

def M_WS(image):
    lb, ub = gethist(image)
    markers = np.zeros(image.shape, dtype=np.uint)
    markers[image < lb] = 1
    markers[image > ub] = 2
    #plt.figure()
    #plt.imshow(markers, cmap="magma")
    binary = segmentation.watershed(image, markers)
    binary[binary == 1] = 0
    binary[binary == 2] = 1
    
    labels = watershedding(binary)
    #plt.figure()
    #plt.imshow(labels, cmap=plt.cm.nipy_spectral)
    return labels


# Markers + Random Walk
def M_RW(image):
    lb, ub = gethist(image)
    markers = np.zeros(image.shape, dtype=np.uint)
    markers[image < lb] = 1
    markers[image > ub] = 2
    #plt.figure()
    #plt.imshow(markers, cmap="magma")
    binary = segmentation.random_walker(image, markers, beta=1000, mode='bf')
    label = ndi.label(ndi.binary_fill_holes(binary - 1))[0]
    #plt.figure()
    #plt.imshow(labels, cmap=plt.cm.nipy_spectral)
    return label
    
label_ws = M_WS(image)
label_rw = M_RW(image)
    
# Random Forest

# VGG16 + RF

# validate segmentation
