# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 18:19:36 2023

@author: Raphael
"""
import Opening
import Segmentation
import Denoise
import Extract
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage import exposure
import pickle
import time

experiment = 3
basedir = Opening.getimfolder(experiment)
info = os.path.join(basedir, "Info.xlsx")
idf = pd.read_excel(info)
print(idf.shape)
tp = []
for strin in idf["Path"].unique():
    if "Deconvolved" in strin:
        tp.append(strin)
idf = idf.drop(idf[~idf["Path"].isin(tp)].index)
print(idf.shape)

dic = {}
maxCID = 0
maxPID = 0
dfs = []
if os.path.exists(os.path.join(basedir,'Cells.csv')) == True:
    oldcdf = pd.read_csv(os.path.join(basedir,'Cells.csv'))
    maxPID = oldcdf["PID"].max()+1
    print(oldcdf.shape)
    dfs.append(oldcdf)
    with open(os.path.join(basedir,'COO.pickle'), 'rb') as handle:
        dic = pickle.load(handle)
    


for PID in idf["PID"].unique():
    if PID < maxPID:
        continue
    start_time = time.time()
    print("PID: "+str(PID)+"/"+str(idf["PID"].max()))
    CL, cdf, cells = Segmentation.segmeasureL(PID,basedir,idf)
    cdf["CID"] = cdf["CID"]+maxCID
    maxCID = cdf["CID"].max()+1
    dic[PID] = {}
    for CID in cdf["CID"].unique():
        dic[PID][CID] = {}
        for ch in ["aSyn_GFP","aSyn_AB","LAMP1","LC3"]:
            if ch not in idf[idf["PID"]==PID]["Channel"].unique():
                continue
            dimg = Denoise.denoise1(idf,cdf,PID,CID,ch,basedir)
            coords = Extract.get_spots(cdf,CID,CL,dimg,30)
            dic[PID][CID][ch] = coords
    with open(os.path.join(basedir,'COO.pickle'), 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dfs.append(cdf)
    new_cdf = pd.concat(dfs,ignore_index=True)
    new_cdf.to_csv(os.path.join(basedir,'Cells.csv'))
    joindf = new_cdf.join(idf.set_index('PID'), on="PID", rsuffix="_")
    print(joindf.groupby(["Group2","Group1"])["CID"].nunique())
    print("--- %s seconds ---" % (time.time() - start_time))
#print(tcdf["VarLap_aSyn_GFP"])
#print(np.unique(CL))






