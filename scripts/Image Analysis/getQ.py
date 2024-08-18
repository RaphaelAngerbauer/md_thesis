# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:24:21 2023

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



ogcdf = pd.read_csv(os.path.join(basedir, "Cells.csv"))

dic = {}


with open(os.path.join(basedir,'COO.pickle'), 'rb') as handle:
    dic = pickle.load(handle)
    


for PID in idf["PID"].unique():
    if PID not in list(dic.keys()):
        continue
    pdf = ogcdf[ogcdf["PID"]==PID]
    start_time = time.time()
    print("PID: "+str(PID)+"/"+str(idf["PID"].max()))
    CL, cdf, cells = Segmentation.segmeasureL(PID,basedir,idf)
    for CID in pdf["CID"].unique():
        sdf = pdf[pdf["CID"]==CID]
        dic[PID][CID]["Q"] = {}
        cimage = CL[int(sdf["bbox-0"].values):int(sdf["bbox-2"].values),int(sdf["bbox-1"].values):int(sdf["bbox-3"].values)]
        counts = np.bincount(cimage.flatten())
        label = np.argmax(counts[1:])+1
        dcimage = np.repeat(cimage.reshape(1,cimage.shape[0],cimage.shape[1]),11,axis=0)

        pmap = np.zeros(dcimage.shape)
        pmap[dcimage==label]=1
        ind = np.reshape(np.indices(pmap.shape).transpose((1,2,3,0)),(1,1,-1,3))[0][0]
        I = np.delete(np.arange(0,ind.shape[0]),np.where(pmap.reshape(1,1,-1)[0][0]==0))
        for c in ["aSyn_GFP","aSyn_AB"]:
            dic[PID][CID]["Q"][c] = {}
            nP = dic[PID][CID][c].shape[0]
            for q in range(1000):
                QI = np.random.choice(I,nP)
                Q = ind[QI]
                dic[PID][CID]["Q"][c][q] = Q
    print("--- %s seconds ---" % (time.time() - start_time))
    
from scipy.spatial import KDTree
def Ct(D,t):
    B = D<t
    return np.mean(B,axis=1)

def getNN(X,Y):
    D = KDTree(X).query(Y)[0]
    return D
    
data = {"Ct":[],"PID":[],"CID":[],"X":[],"Y":[],"t":[],"Group":[],"Xc":[],"Yc":[],"p05":[],"p01":[],"p001":[]}
for key in dic:
    PID = key
    G1 = idf[idf["PID"]==PID]["Group1"].unique()[0]
    xlen = idf["xlen"][idf["PID"]==PID].unique()[0]
    ylen = idf["ylen"][idf["PID"]==PID].unique()[0]
    zlen = idf["zlen"][idf["PID"]==PID].unique()[0]
    V = np.array([[zlen,ylen,xlen]])
    for k2 in dic[key]:
        CID = k2
        X = []
        Xn = ""
        if "LAMP1" in list(dic[key][k2].keys()):
            X = dic[key][k2]["LAMP1"]
            Xn = "LAMP1"
        if "LC3" in list(dic[key][k2].keys()):
            X = dic[key][k2]["LC3"]
            Xn = "LC3"
        Xc = X.shape[0]
        X = np.multiply(X,np.repeat(V,X.shape[0],axis=0))
        for aSyn in ["aSyn_GFP","aSyn_AB"]:
            QD = []
            for k3 in dic[key][k2]["Q"][aSyn]:
                
                Y = dic[key][k2]["Q"][aSyn][k3]
                
                Y = np.multiply(Y,np.repeat(V,Y.shape[0],axis=0))
                QD.append(getNN(X,Y))
            Y = dic[key][k2][aSyn]
            Yc = Y.shape[0]
            Y = np.multiply(Y,np.repeat(V,Y.shape[0],axis=0))
            D = getNN(X,Y)
            for t in np.arange(0,5,0.1):
                C = Ct(np.array(D).reshape(1,-1),t)
                Q = Ct(np.array(QD),t)
                Q05 = np.percentile(Q,95)
                Q01 = np.percentile(Q,99)
                Q001 = np.percentile(Q,99.9)
                data["Ct"].append(C[0])
                data["PID"].append(PID)
                data["CID"].append(CID)
                data["X"].append(Xn)
                data["Y"].append(aSyn)
                data["t"].append(t)
                data["Group"].append(G1)
                data["Xc"].append(Xc)
                data["Yc"].append(Yc)
                data["p05"].append(Q05)
                data["p01"].append(Q01)
                data["p001"].append(Q001)
        print(CID)
            
df = pd.DataFrame(data=data)
df.to_csv(os.path.join(basedir,'SimpleNN.csv'))   
    
    
    
    
    
    
    
    
    