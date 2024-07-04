# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:16:12 2023

@author: Raphael
"""

import numpy as np
from findmaxima2d import find_maxima, find_local_maxima
from skimage import exposure

def get_spots(cdf,CID,CL,dimg,ntol):
    coords = []
    r = cdf[cdf["CID"] == CID]
    cCL = CL[int(r["bbox-0"].values):int(r["bbox-2"].values),int(r["bbox-1"].values):int(r["bbox-3"].values)]
    label = r["label"].values[0]
    for st in range(dimg.shape[0]):
        pimp = exposure.rescale_intensity(dimg[st,:,:],out_range="uint8")
        pimp[cCL != label] = 0
        local_max = find_local_maxima(pimp)
        y, x, regs = find_maxima(pimp,local_max,ntol)
        s = len(x)*[st]
        coords.append(np.array([s,y,x]).T)
    return np.concatenate(coords)