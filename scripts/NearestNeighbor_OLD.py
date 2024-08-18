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
    
    def D(self,X,Y):
        D = KDTree(X).query(Y)[0]
        return D
    
    def Random_Points_in_Bounds(self,ROI, number):   
        x = np.random.uniform( ROI.left, ROI.right, number )
        y = np.random.uniform( ROI.top, ROI.bottom, number )
        return x, y
    
    def Q(self,X,CID):
        ROI = None
        for R in roiread(self.df["ROI_path"][self.df["CID"]==CID].values[0]):
            if R.name == "Cell_"+str(CID):
                ROI = R
        gdf_poly = gpd.GeoDataFrame(index=["ROI"], geometry=[Polygon(ROI.coordinates())])
        x,y = self.Random_Points_in_Bounds(ROI, self.ranN)
        df = pd.DataFrame()
        df['points'] = list(zip(x,y))
        df['points'] = df['points'].apply(Point)
        gdf_points = gpd.GeoDataFrame(df, geometry='points')
        Sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, predicate="within", how='left')
        Qpoints = gdf_points[Sjoin.index_right=='ROI']
        XY = np.array([QP[0].coords[0] for QP in Qpoints.values])
        QP = []
        for i in range(self.Zcount):
            z = np.full(Qpoints.values.shape[0],i*self.zlen)
            QP.append(np.vstack([(XY.T)*self.xlen,z]).T)
        Xxy = X[:,0:2].T-np.array([[ROI.left*self.xlen],[ROI.bottom*self.xlen]])
        Q = KDTree(np.vstack([Xxy+np.array([[ROI.left*self.xlen],[ROI.top*self.xlen]]),X[:,2].T]).T).query(np.concatenate(QP))[0]
        return Q
    
    def cell_NN(self,CID,data):
        pdf = pd.read_csv(self.df["coord_path"][self.df["CID"]==CID].values[0],sep="\t")
        pdf = pdf[pdf["InCytoplsm"]==True]
        X = None
        Y = {}
        for channel in pdf["channel"].unique():
            tempdf = pdf[pdf["channel"]==channel]
            if channel == self.XChannel:
                X = np.array([tempdf["x"].values*self.xlen,tempdf["y"].values*self.ylen,tempdf["z"].values*self.zlen]).T
            else:
                Y[channel]=np.array([tempdf["x"].values*self.xlen,tempdf["y"].values*self.ylen,tempdf["z"].values*self.zlen]).T
        self.dic[CID]={}
        self.dic[CID]["D"]={}
        for key in Y:
            self.dic[CID]["D"][key]=self.D(X,Y[key])
        self.dic[CID]["Q"]=self.Q(X,CID)
        data["CID"].append(CID)
        data["N_X"].append(X.shape[0])
        return data
    
    def calculate(self):
        data = {}
        C = ["N_X","N_D","N_Q"]
        for h in ["CID","N_X"]:
            data[h]=[]
            if h != "CID" and h in self.df.columns:
                self.df.drop(h,axis=1)
        if "N_D" in list(self.df.columns):
            self.df = self.df.drop(C, axis=1)
        for CID in tqdm(self.df["CID"].values):
            data = self.cell_NN(CID,data)
        self.df = self.df.merge(pd.DataFrame(data=data),how="left",on="CID")
        self.df.to_excel(os.path.join(self.basedir,"Results.xlsx"), index=False)
        return self.dic
        
    def save(self,path=None):
        if path == None:
            path=os.path.join(self.basedir,"NN.p")
        pickle.dump(self.dic, open(path, "wb"))
        print("Saved at: "+path)
        return
    
    def load(self,path=None):
        if path == None:
            path=os.path.join(self.basedir,"NN.p")
        self.dic = pickle.load(open(path, "rb"))
        print("Opened from: "+path)
        return
    
    def C(self,D,N):
        return D/N

    def e(self,Ct,Co):
        #print(Ct)
        Ct = Ct.replace(to_replace = 0, value = 0.00000001)
        Co = Co.replace(to_replace = 0, value = 0.00000001)
        return np.log(Ct/(1-Ct))-np.log(Co/(1-Co))  

    def crit_e(self,N,Ct,Co):
        CC = binom.ppf(0.95, N, Ct)
        return self.e(self.C(CC,N),Co)
    
    def p_val_step(self,e,N,CtX,CoX):
        E = np.exp(e+np.log(CoX/(1-CoX)))
        Ct = E/(1+E)
        p = 1-binom.cdf(Ct*N, N, CtX)
        return p

    def confidence(self,N,Ct,Co):
        CI_Ct = binom.interval(0.95,N,Ct)
        CI = [self.e(ci/N,Co) for ci in CI_Ct]
        return CI[0],CI[1]
    
    def get_pval(self,t=None,test=None,control=None):
        if control != None:
            self.control = control
        if t != None:
            self.threshold = t
        data = {}
        C = ["N_D","N_Q","D","Q","radius"]
        for h in ["CID","N_D","N_Q","D","Q","radius"]:
            data[h]=[]
        for c in C:
            if c in list(self.df.columns):
                self.df = self.df.drop(c, axis=1)
        for CID in self.df["CID"].unique():
            data["CID"].append(CID)
            data["radius"].append(self.threshold)
            data["D"].append(np.sum(self.dic[CID]["D"][self.YChannel]<self.threshold))
            data["Q"].append(np.sum(self.dic[CID]["Q"]<self.threshold))
            data["N_D"].append(len(self.dic[CID]["D"][self.YChannel]))
            data["N_Q"].append(len(self.dic[CID]["Q"]))
        self.df = self.df.merge(pd.DataFrame(data=data),how="left",on="CID")
        df2 = self.df.groupby(["Group2","Group1"])["N_D","N_X","N_Q","D","Q"].sum().reset_index()
        df2["Ct"] = self.C(df2["D"],df2["N_D"])
        df2["Co"] = self.C(df2["Q"],df2["N_Q"])
        df2["e"] = self.e(df2["Ct"],df2["Co"])
        df2["crit_e"] = self.crit_e(df2["N_D"],df2["Ct"],df2["Co"])
        df2["CtX"] = df2.apply(lambda row: pd.Series([df2["Ct"][df2["Group2"]==row["Group2"]][df2["Group1"]==self.control].values[0]]), axis=1)
        df2["CoX"] = df2.apply(lambda row: pd.Series([df2["Co"][df2["Group2"]==row["Group2"]][df2["Group1"]==self.control].values[0]]), axis=1)
        df2["p"] = self.p_val_step(df2["e"],df2["N_D"],df2["CtX"],df2["CoX"])
        df2["ci_lo"],df2["ci_hi"] = self.confidence(df2["N_D"],df2["Ct"],df2["Co"])
        self.df2 = df2
        df2.groupby(["Group2","Group1"])
        return df2
    
    def get_pval_indiv(self,t=None,control=None):
        if control != None:
            self.control = control
        if t != None:
            self.threshold = t
        data = {}
        C = ["N_D","N_Q","D","Q","radius"]
        for h in ["CID","N_D","N_Q","D","Q","radius"]:
            data[h]=[]
        for c in C:
            if c in list(self.df.columns):
                self.df = self.df.drop(c, axis=1)
        for CID in self.df["CID"].unique():
            data["CID"].append(CID)
            data["radius"].append(self.threshold)
            data["D"].append(np.sum(self.dic[CID]["D"][self.YChannel]<self.threshold))
            data["Q"].append(np.sum(self.dic[CID]["Q"]<self.threshold))
            data["N_D"].append(len(self.dic[CID]["D"][self.YChannel]))
            data["N_Q"].append(len(self.dic[CID]["Q"]))
        self.df = self.df.merge(pd.DataFrame(data=data),how="left",on="CID")
        df = self.df
        df["Ct"] = self.C(df["D"],df["N_D"])
        df["Co"] = self.C(df["Q"],df["N_Q"])
        df["e"] = self.e(df["Ct"],df["Co"])
        df["crit_e"] = self.crit_e(df["N_D"],df["Ct"],df["Co"])
        df["CtX"] = df.apply(lambda row: pd.Series([df["Ct"][df["Group2"]==row["Group2"]][df["Group1"]==self.control].values[0]]), axis=1)
        df["CoX"] = df.apply(lambda row: pd.Series([df["Co"][df["Group2"]==row["Group2"]][df["Group1"]==self.control].values[0]]), axis=1)
        df["p"] = self.p_val_step(df["e"],df["N_D"],df["CtX"],df["CoX"])
        df["ci_lo"],df["ci_hi"] = self.confidence(df["N_D"],df["Ct"],df["Co"])
        df.groupby(["Group2","Group1"])
        return df
    
    def get_NN_per_cell(self,t=0.5):
        data={"CID":[],"D":[],"Q":[]}
        for key in self.dic:
            data["CID"].append(key)
            data["D"].append(np.sum(data[key]["D"]<t))
            data["Q"].append(np.sum(data[key]["D"]<t))
        df = self.df.merge(pd.DataFrame(data=data),how="left",on="CID")
        return df
    
    
        
    
        