# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 17:56:50 2023

@author: Raphael
"""

import numpy as np
import tifffile as tiff
from roifile import ImagejRoi
from skimage.measure import grid_points_in_poly

def getImage(path,cn):
    #Implements opening of a specified image
    #Inputs:
        #path: path to the image
        #ch: channel name of the image
    #First read where the image is stored
    cID = cn
    with tiff.TiffFile(path) as image:
        #read image as dask array -° speeds up computation
        img = tiff.imread(path)
        if len(img.shape)==4:
            img = img[:,cID,:,:]
            return np.array(img)
        else:
            img = img[cID,:,:]
            return np.array(img)
        #return as np array
        return np.array(img)
            
def crop(image,ROI):
    #Implements cropping of a 2d image
    #Inputs:
        #image: image to be cropped
        #ROI: list of coordinates specifying the cell boundaries
    #cropping of the image by reading out the bounding box information
    if len(image.shape)==3:
        cimage = image[:,int(ROI.top):int(ROI.bottom),int(ROI.left):int(ROI.right)]
        return cimage
    else:
        cimage = image[int(ROI.top):int(ROI.bottom),int(ROI.left):int(ROI.right)]
        return cimage
    
def get_cropped(df,CID,cn):
    path = df["path"][df["CID"]==CID].values[0]
    roi_path = df["ROI_path"][df["CID"]==CID].values[0]
    img = getImage(path,cn)
    ROI = None
    ROIs = ImagejRoi.fromfile(roi_path)
    for r in ROIs:
        if r.name == "Cell_"+str(CID):
            ROI = r
    return crop(img,ROI)

def get_cropped_denoised(df,CID,cn):
    path = df["denoised_path"][df["CID"]==CID].values[0]
    roi_path = df["ROI_path"][df["CID"]==CID].values[0]
    img = getImage(path,cn)
    ROI = None
    ROIs = ImagejRoi.fromfile(roi_path)
    for r in ROIs:
        if r.name == "Cell_"+str(CID):
            ROI = r
    return crop(img,ROI)

def get_mask(df,CID,shape):
    roi_path = df["ROI_path"][df["CID"]==CID].values[0]
    ROI = None
    ROIs = ImagejRoi.fromfile(roi_path)
    for r in ROIs:
        if r.name == "Cell_"+str(CID):
            ROI = r
    m = grid_points_in_poly(shape,np.subtract(ROI.coordinates(),np.array([[ROI.left,ROI.top]])))
    return m

def get_ROI(df,CID):
    roi_path = df["ROI_path"][df["CID"]==CID].values[0]
    ROIs = ImagejRoi.fromfile(roi_path)
    for r in ROIs:
        if r.name == "Cell_"+str(CID):
            return r