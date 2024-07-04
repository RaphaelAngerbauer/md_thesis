# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:09:43 2023

@author: Raphael
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from roifile import ImagejRoi
import tifffile as tiff
from scipy import ndimage as ndi
from tqdm import tqdm
import time

from skimage.filters import gaussian,threshold_multiotsu,threshold_otsu
from skimage.restoration import rolling_ball
from skimage.segmentation import watershed,clear_border
from skimage.exposure import equalize_adapthist,rescale_intensity
from skimage.measure import regionprops_table, find_contours,blur_effect, points_in_poly,grid_points_in_poly
from skimage.feature import peak_local_max
from skimage.util import img_as_ubyte


from findmaxima2d import find_maxima, find_local_maxima

basedir = "D:/Raphael/Data/7. Exocytosed aSyn-GFP gets taken up by untransfected cells"
df = pd.read_excel(os.path.join(basedir,"Results.xlsx"))
ch_dic={"aSyn-GFP":0,"aSyn-AB":1,"LAMP1/LC3":2}
channel_aut = "LAMP1/LC3"
channel_aSyn = "aSyn-GFP"
channel_AB = "aSyn-AB"


def getImage(path,ch="",cn=None):
    #Implements opening of a specified image
    #Inputs:
        #path: path to the image
        #ch: channel name of the image
    #First read where the image is stored
    cID = cn
    with tiff.TiffFile(path) as image:
        #read image as dask array -° speeds up computation
        img = tiff.imread(path)
        if cn == None:
            #get the arrangement of desired channel
            C = image.imagej_metadata["channel"]
            I = C.split(",")
            cID = None
            for i in I:
                if ch in i:
                    cID = int(i.split(":")[1].replace(" ","").replace("}",""))
            #determine the number of the channel
        if len(img.shape)==4:
            img = img[:,cID,:,:]
            return np.array(img)
        else:
            img = img[cID,:,:]
            return np.array(img)
        #return as np array
        return np.array(img)
    
def change_path(path, old="E", new="D"):
    return new+path[1::]

def detect_objects(PID,tol=30):
    pdf = df[df["PID"] == PID]
    path = pdf["path"].unique()[0]
    path = change_path(path)
    img = img_as_ubyte(rescale_intensity(gaussian(getImage(path,cn=0).astype("uint16"),sigma=10,preserve_range=True)))
    local_max = find_local_maxima(img)
    RD = {"CID":[]}
    RD["PointCount"] = []
    RD["int"] = []
    y, x, regs = find_maxima(img,local_max,tol)
    coo = np.array([y,x])
    roipath = pdf["ROI_path"].unique()[0]
    roipath = change_path(roipath)
    ROIs = ImagejRoi.fromfile(roipath)
    for ROI in ROIs:
        RD["CID"].append(int(ROI.name.split("_")[1]))
        pip = points_in_poly(coo.T,ROI.coordinates())
        RD["PointCount"].append(np.sum(pip))
        if np.sum(pip) > 0:

            RD["int"].append(np.sum(img[tuple(coo)]*pip)/np.sum(pip))
        else:
            RD["int"].append("NaN")
    pdf = pdf.merge(pd.DataFrame(data=RD), on='CID', how='outer')
    return pdf


test = detect_objects(1178,tol=80)
print(test["PointCount"].sum())

pdfs = []
for PID in tqdm(df["PID"].unique()):
    
    pdf = detect_objects(PID,tol=100)
    pdfs.append(pdf)
   
pd.concat(pdfs).to_excel(os.path.join(basedir,"Results_2.xlsx"))
    

