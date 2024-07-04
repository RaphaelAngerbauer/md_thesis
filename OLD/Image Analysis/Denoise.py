# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:41:47 2023

@author: Raphael
"""
from skimage.restoration import rolling_ball
from skimage.feature import peak_local_max
from skimage.filters import laplace, gaussian
import pandas as pd
import numpy as np
import Opening

def crop(image, label, df):
    r = df[df["label"] == label]
    cimage = image[:,int(r["bbox-0"].values):int(r["bbox-2"].values),int(r["bbox-1"].values):int(r["bbox-3"].values)]
    return cimage

def reversecrop(zeros, insert, label, df):
    r = df[df["label"] == label]
    zeros[int(r["bbox-0"].values):int(r["bbox-2"].values),int(r["bbox-1"].values):int(r["bbox-3"].values)] = insert
    return zeros

def denoise1(idf,cdf,PID,CID,ch,basedir):
    img = Opening.getImage(idf, PID,ch, basedir)
    label = cdf["label"][cdf["CID"]==CID].values[0]
    newimg = []
    cimg = crop(img, label, cdf)
    for st in range(img.shape[0]):
        gimg = gaussian(cimg[st,:,:],sigma=0.2)
        bg = rolling_ball(gimg, radius = 15)
        bg = gaussian(bg,sigma=2)
        bgsub = gimg-bg
        newimg.append(bgsub)
    return np.array(newimg)