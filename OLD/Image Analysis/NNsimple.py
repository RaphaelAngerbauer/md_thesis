# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:39:21 2023

@author: Raphael
"""


import numpy as np
import pickle
import Opening
import os
from scipy.spatial import KDTree
import pandas as pd


def Ct(D,t):
    B = D<t
    return B.sum()/D.shape[0]

def getNN(X,Y):
    D = KDTree(X).query(Y)[0]
    return D


experiment = 3
basedir = Opening.getimfolder(experiment)
print(basedir)
info = os.path.join(basedir, "Info.xlsx")
idf = pd.read_excel(info)

coords = {}

with open(os.path.join(basedir,'COO.pickle'), 'rb') as handle:
    coords = pickle.load(handle)
    

data = {"Ct":[],"PID":[],"CID":[],"X":[],"Y":[],"t":[],"Group":[],"Xc":[],"Yc":[]}
for key in coords:
    PID = key
    G1 = idf[idf["PID"]==PID]["Group1"].unique()[0]
    xlen = idf["xlen"][idf["PID"]==PID].unique()[0]
    ylen = idf["ylen"][idf["PID"]==PID].unique()[0]
    zlen = idf["zlen"][idf["PID"]==PID].unique()[0]
    V = np.array([[zlen,ylen,xlen]])
    for k2 in coords[key]:
        CID = k2
        X = []
        Xn = ""
        if "LAMP1" in list(coords[key][k2].keys()):
            X = coords[key][k2]["LAMP1"]
            Xn = "LAMP1"
        if "LC3" in list(coords[key][k2].keys()):
            X = coords[key][k2]["LC3"]
            Xn = "LC3"
        Xc = X.shape[0]
        X = np.multiply(X,np.repeat(V,X.shape[0],axis=0))
        for aSyn in ["aSyn_GFP","aSyn_AB"]:
            
            Y = coords[key][k2][aSyn]
            Yc = Y.shape[0]
            Y = np.multiply(Y,np.repeat(V,Y.shape[0],axis=0))
            D = getNN(X,Y)
            for t in np.arange(0,5,0.1):
                C = Ct(D,t)
                data["Ct"].append(C)
                data["PID"].append(PID)
                data["CID"].append(CID)
                data["X"].append(Xn)
                data["Y"].append(aSyn)
                data["t"].append(t)
                data["Group"].append(G1)
                data["Xc"].append(Xc)
                data["Yc"].append(Yc)
        print(CID)
            
df = pd.DataFrame(data=data)
df.to_csv(os.path.join(basedir,'SimpleNN.csv'))