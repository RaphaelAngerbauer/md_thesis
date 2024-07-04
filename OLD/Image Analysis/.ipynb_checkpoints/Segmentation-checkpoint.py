# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:30:55 2023

@author: Raphael
"""
from skimage.filters import threshold_multiotsu
import numpy as np
from numpy.matlib import repmat
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import regionprops, regionprops_table
from skimage.restoration import rolling_ball
from skimage.feature import peak_local_max
from skimage.filters import laplace, gaussian
from skimage import morphology

from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize

import Opening

def measure_median(label, intensity):
    return np.median(intensity[label])

def rescale_before(img, pixlenimg):
    from skimage.transform import rescale
    pixlensd = 0.3488
    scaling_factor = pixlenimg/pixlensd
    image_rescaled = rescale(img, scaling_factor, anti_aliasing=True, preserve_range = True)
    return image_rescaled

def rescale_after(img, dims):
    from skimage.transform import resize
    image_resized = resize(img, dims, preserve_range=True, order=0, anti_aliasing=False).astype('uint8')
    return image_resized

def segment_cells(cells, labels):
    t = threshold_multiotsu(cells)
    minimg = np.min(cells)
    binary = cells > (2*t[0]+minimg)/3
    binary = ndi.binary_fill_holes(binary)
    cellL = watershed(-cells, markers = labels, mask = binary)
    return cellL

def segment_slice(NDID,basedir,idf,xlen, model,CytoplasmChannel):
    DAPIimg = Opening.getImage(idf, int(idf["ID"][idf["NDID"]==NDID][idf["Channel"]=="DAPI"].values[0]), basedir)
    if model == None:
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    img_rescaled_DAPI = rescale_before(DAPIimg, xlen)
    labelssd, _ = model.predict_instances(normalize(img_rescaled_DAPI),n_tiles=(10,10))
    labels_rescaled = rescale_after(labelssd, DAPIimg.shape)
    nucleidf = pd.DataFrame(regionprops_table(labels_rescaled,intensity_image=DAPIimg,properties = ('label', 'bbox',"intensity_mean"), extra_properties=(measure_median,)))
    del DAPIimg
    CELLimg = Opening.getImage(idf, int(idf["ID"][idf["NDID"]==NDID][idf["Channel"]==CytoplasmChannel].values[0]), basedir)
    cells = segment_cells(rescale_before(CELLimg, xlen),labelssd)
    cellL = rescale_after(cells, CELLimg.shape)
    del CELLimg
    return labels_rescaled, cellL,nucleidf

def segmeasure(NDID,basedir,idf,model,CytoplasmChannel):
    xlen = idf["xlen"][idf["NDID"]==NDID].unique()[0]
    NL, CL, tcdf = segment_slice(NDID,basedir,idf,xlen, model,CytoplasmChannel)
    for ch in idf["Channel"].unique():
        if ch in ["DAPI"]:
            continue
        intimg = Opening.getImage(idf, int(idf["ID"][idf["NDID"]==NDID][idf["Channel"]==ch].values[0]), basedir)
        celldf = pd.DataFrame(regionprops_table(CL,intensity_image=intimg,properties = ('label', 'bbox', "intensity_max","intensity_min","intensity_mean"), extra_properties=(measure_median,)))
        tcdf.join(celldf, on = "label", lsuffix = "DAPI",rsuffix = ch)
    tcdf["NDID"] = NDID
    tcdf["CID"] = tcdf["label"]
    return NL, CL, tcdf

def segment_cellsL(PID,basedir,idf,xlen,CytoplasmChannel):
    cellst = np.array(Opening.getImage(idf, PID,CytoplasmChannel, basedir))
    cells = np.max(cellst,axis=0)
    t = threshold_multiotsu(cells)
    print(t)
    binary1 = cells >= t[0]
    binary1 = ndi.binary_fill_holes(binary1)
    binary = np.zeros(binary1.shape, dtype = np.uint8)
    binary[binary1 == True] = 1
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, footprint=np.ones((500, 500)), labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    cellL = watershed(-distance, markers = markers, mask = binary)
    cellL = morphology.remove_small_objects(cellL, 100000)
    tcdf = pd.DataFrame(data = {"label": np.unique(cellL)})
    return cellL, tcdf, cellst

def segmeasureL(PID,basedir,idf):
    xlen = idf["xlen"][idf["PID"]==PID].unique()[0]
    CL, tcdf, cells = segment_cellsL(PID,basedir,idf,xlen,"aSyn_GFP")
    for ch in idf["Channel"][idf["PID"]==PID].unique():
        intimg = np.max(np.array(Opening.getImage(idf, PID, ch, basedir)),axis=0)
        celldf = pd.DataFrame(regionprops_table(CL,intensity_image=intimg,properties = ('label', 'bbox', "intensity_max","intensity_min","intensity_mean"), extra_properties=(measure_median,)))
        tcdf = tcdf.join(celldf, on = "label",rsuffix = "_"+ch,how= "right")
    tcdf["label"] = np.unique(CL)[1:]
    tcdf["PID"] = PID
    tcdf["CID"] = tcdf["label"]
    return CL, tcdf, cells
    
    
    