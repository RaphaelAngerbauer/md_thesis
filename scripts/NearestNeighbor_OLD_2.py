# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:15:36 2023

@author: Raphael
"""

from scipy.spatial import KDTree
import numpy as np
import pandas as pd
from roifile import roiread
from shapely.geometry import Point, Polygon
import geopandas as gpd
import os
from tqdm import tqdm
import pickle
    
from scipy.stats import binom

from skimage.measure import points_in_poly


class NN(object):
    
    def __init__(self,basedir = None,data=None,exID=3,df = pd.DataFrame(),Zcount=10,xlen=0.07530842430163,ylen=0.07530842430163,zlen=0.2,ranN=100_000,XChannel="LAMP1/LC3",YChannel="aSyn-GFP",threshold=0.3,control="X"):
        #if basedir is not specified it is calculated from exID and/or data
        self.df2 = pd.DataFrame()
        self.control = control
        self.dic = {}
        self.Zcount=Zcount
        self.xlen=xlen
        self.ylen=ylen
        self.zlen=zlen
        self.XChannel=XChannel
        self.YChannel=YChannel
        self.ranN = ranN
        self.threshold = threshold
        if basedir==None:
            if data == None:
                for folder in os.listdir(data):
                    if str(exID) in folder:
                        self.basedir = os.path.join("Data", folder)
            else:
                for folder in os.listdir(data):
                    if str(exID) in folder:
                        self.basedir = os.path.join(data, folder)
        else: 
            self.basedir=basedir
        if df.empty == True:
            resultspath = os.path.join(self.basedir,"Results.xlsx")
            if os.path.exists(resultspath) == True:
                self.df = pd.read_excel(resultspath)
        
        return
    
    def D(self,pdf):
        ppdf = pdf[pdf["Channel"]==self.YChannel]
        Ny = ppdf["InAutophagosome"].count()
        D = 0
        if Ny > 0:
            D = ppdf[ppdf["InAutophagosome"]==True]["InAutophagosome"].count()/Ny
        return D,Ny
    
    def Random_Points_in_Bounds(self,ROI, number):   
        x = np.random.uniform( ROI.left, ROI.right, number )
        y = np.random.uniform( ROI.top, ROI.bottom, number )
        return x, y
    
    def Q(self,zsize=11):
        Q = self.df["Autophagosome_area"]/(self.df["area_GFP"]*zsize)
        return Q
    
    
    
    def C(self,D,N):
        return D/N

    def e(self,Ct,Co):
        #print(Ct)
        Ct = Ct.replace(to_replace = 0, value = 0.00000001)
        Co = Co.replace(to_replace = 0, value = 0.00000001)
        Ct = Ct.replace(to_replace = 1, value = 0.9999999)
        Co = Co.replace(to_replace = 1, value = 0.9999999)
        return np.log(Ct/(1-Ct))-np.log(Co/(1-Co))  

    def crit_e(self,N,Ct,Co):
        CC = binom.ppf(0.95, N, Ct)
        return self.e(self.C(CC,N),Co)
    
    def p_val_step(self,e,N,CtX,CoX):
        E = np.exp(e+np.log(CoX/(1-CoX)))
        Ct = E/(1+E)
        p = 1-binom.cdf(Ct*N, N, CtX)
        return p
    
    def calculate_e(self):
        for col in ["Nx","Ny","Ct","Cq","e"]:
            if col in self.df.columns:
                self.df = self.df.drop(col,axis=1)
        data = {"CID":[],"Nx":[],"Ny":[],"Ct":[],"st":[]}
        for CID in self.df["CID"].unique():
            data["CID"].append(CID)
            pdf = pd.read_csv(self.df["coord_path"][self.df["CID"]==CID].values[0],sep="\t")
            pdf = pdf[pdf["CID"]==CID]
            pdf = pdf[pdf["InCytoplasm"]==True]
            data["Nx"].append(pdf[pdf["Channel"]==self.XChannel]["InAutophagosome"].count())
            data["st"].append(len(pdf[pdf["Channel"]==self.XChannel]["z"].unique()))
            Ct,Ny = self.D(pdf)
            data["Ct"].append(Ct)
            data["Ny"].append(Ny)
        self.df = pd.concat([self.df,pd.DataFrame(data=data)],axis=1)
        self.df["Cq"] = self.Q(self.df["st"])
        self.df["e"] = self.e(self.df["Ct"],self.df["Cq"])
        self.df.to_excel(os.path.join(self.basedir,"Results.xlsx"),index=False)
        return
    
    
        
    
        