# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:19:18 2023

@author: Raphael
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,cdist,squareform
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from scipy.optimize import fmin
import math
from roifile import ImagejRoi
from shapely.geometry import Point, Polygon
import geopandas as gpd
from numpy.random import randint,normal
from cmaes import CMA

class NearestNeighbour():
    
    def __init__(self,df,pdf,x,y,ROI):
        self.nknots = 21
        self.df = df
        self.pdf = pdf
        self.ROI = ROI
        self.xTree = {}
        self.d = {}
        self.q = {}
        self.D = {}
        self.shape = self.f_hermquist
        for CID in self.df["CID"].unique():
            cdf = self.pdf[self.pdf["CID"]==CID]
            self.xTree[CID] = KDTree(np.array([cdf["x"][cdf["Channel"]==x].values,cdf["y"][cdf["Channel"]==x].values]).T)
            self.q[CID],self.d[CID] = self.get_q(CID)
            self.D[CID],_ = self.xTree[CID].query(np.array([cdf["x"][cdf["Channel"]==y].values,cdf["y"][cdf["Channel"]==y].values]).T)
        
        
    def sample_q(self,CID,number = 10000):
        ROI = self.ROI[CID]
        gdf_poly = gpd.GeoDataFrame(index=["ROI"], geometry=[Polygon(ROI.coordinates())])
        x = np.random.uniform( ROI.left, ROI.right, number )
        y = np.random.uniform( ROI.top, ROI.bottom, number )
        df = pd.DataFrame()
        df['points'] = list(zip(x,y))
        df['points'] = df['points'].apply(Point)
        gdf_points = gpd.GeoDataFrame(df, geometry='points')
        Sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, predicate="within", how='left')
        Qpoints = gdf_points[Sjoin.index_right=='ROI'].values
        Q = []
        for pp in Qpoints:
            Q.append([pp[0].x, pp[0].y])
        Q = np.array(Q)
        return Q
    
    def get_q(self,CID):
        sample_points = self.sample_q(CID)
        qD,_ = self.xTree[CID].query(sample_points)
        q_of_d = gaussian_kde(qD)
        d = np.linspace(np.min(qD),np.max(qD),250)
        q = q_of_d(d)
        return q,d
    
    def get_p_of_d(self,params,CID):
        d = self.d[CID]
        q = self.q[CID]
        D = self.D[CID]
        # adjust parameters
        epsilon = params[0]
        sigma = params[1]
        #epsilon can be  max 10 in magnitude
        if abs(epsilon) > 10:
            epsilon = 10*np.sign(epsilon)
        #sigma must not be too small and must be positive
        sigma = abs(sigma)
        if sigma <= 0.001:
            sigma = 0.001
        #calculate Z
        g_of_r = np.exp(-epsilon*self.shape(d/sigma))
        support = g_of_r*q
        diff = np.diff(d)
        integrand = (support[:-1]+support[1:])/2
        Z = np.sum(integrand*diff)
        #calculate P(d)
        P = g_of_r*q/Z
        p_of_d = np.interp(D,d,P)
        return p_of_d
    
    def negloglik(self,epsilon,sigma,CID):
        params = [epsilon,sigma]
        p_of_d = self.get_p_of_d(params,CID)
        return -np.sum(np.log(p_of_d))
    
    def pooled_negloglik_nested(self,sigma):
        nll = 0
        for CID in self.df["CID"].unique():
            epsilon_0 = 1
            optimized = fmin(self.negloglik,epsilon_0,args=(sigma,CID),full_output=True,disp=False)
            nll = nll+optimized[1]
        return nll
    
    def get_parametric(self,nested=True):
        sigma_0 = 1
        sigma = 0
        if nested == True:
            optimized = fmin(self.pooled_negloglik_nested,sigma_0,disp=False)
            sigma = optimized[0]
        E = []
        for CID in self.df["CID"].values:
            epsilon_0 = 1
            optimized = fmin(self.negloglik,epsilon_0,args=(sigma,CID),disp=False)
            E.append(optimized[0])
        self.df["e"] = E
        self.df["sigma"] = sigma
        return self    

    def get_nonparametric(self,nknots=21,s=2):
        self.nknots = nknots
        self.s = s
        params_0 = np.zeros(self.nknots)
        
        optimized = fmin(self.penalized_joint_loglik,params_0)
        self.weights = optimized[0]
        return 
    
    def show_np_phi(self):
        self.get_nonparametric()
        d = np.linspace(-5,95,self.nknots)
        phi = self.phi_np(d,self.weights)
        plt.plot(d,phi)
        return
    
    def penalized_joint_loglik(self,params):
        nll = 0
        penalty = np.sum(np.square(np.diff(params)/self.s))
        for CID in self.df["CID"].unique():
            p_of_d = self.np_p_of_d(params,CID)
            nllt = np.sum(np.log(p_of_d))
            nll = nll+nllt
        
        return -nll-penalty
    
    def phi_np(self,d,params):
        return np.sum(self.piecewise_linear(d)*params.reshape(-1,1),axis = 0)
    
    def np_p_of_d(self,params,CID):
        d = self.d[CID]
        q = self.q[CID]
        D = self.D[CID]
        #calculate Z
        phi = self.phi_np(d,params)
        g_of_r = np.exp(-phi)
        support = g_of_r*q
        diff = np.diff(d)
        integrand = (support[:-1]+support[1:])/2
        Z = np.sum(integrand*diff)
        #calculate P(d)
        P = g_of_r*q/Z
        p_of_d = np.interp(D,d,P)
        return p_of_d
    
    def piecewise_linear(self,d):
        knots = np.linspace(-5,95,self.nknots).reshape(-1,1)
        K = np.tile(knots,(1,d.size))
        spacing = knots[1]-knots[0]
        d = d.reshape(1,-1)
        if d.shape[1] < d.shape[0]:
            d = d.T
            
        D = np.tile(d,(self.nknots,1))
        f = abs(D-K)
        f[f>spacing] = 0
        f = 1-(f/spacing)
        return f
    
    def extimate_phi(self,CID):
        d = self.d[CID]
        q = self.q[CID]
        D = self.D[CID]
        p = gaussian_kde(D)
        P = p(d)
        phi = -np.log(P/q)
        test = q*np.exp(-phi)
        plt.plot(d,phi)
        
    
    def f_hermquist(self,d):
        f = -(1-d)
        f[d>=0] = -1/(d[d>=0]+1)
        return f
        
ROIdic = {}  
pdf = pd.DataFrame()
CIDs = []
xmax = randint(50,200)
ymax = randint(50,200)
ROIx = np.linspace(1,xmax,xmax)
ROIy = np.linspace(1,ymax,ymax)
mx,my = np.meshgrid(ROIx,ROIy)
sx = np.concatenate((mx[0],mx.T[-1],np.flip(mx[0]),mx.T[0])) 
sy = np.concatenate((my[0],my.T[-1],my[-1],np.flip(my.T[0]))) 
#plt.plot(sx,sy)
testROI = ImagejRoi.frompoints(np.array([sx,sy]).T)
for i in range(3):
    
    ROIdic[i] = testROI
    shape = (randint(200,800),2)
    x = randint(1, high=[xmax,ymax], size=shape)
    xdf = pd.DataFrame()
    xdf["x"] = x.T[0]
    xdf["y"] = x.T[1]
    xdf["Channel"] = "x"
    y = normal(loc=0,scale=10^(2),size=shape)
    y = y+x
    #y = randint(1, high=[xmax,ymax], size=shape)
    ydf = pd.DataFrame()
    ydf["x"] = y.T[0]
    ydf["y"] = y.T[1]
    ydf["Channel"] = "y"
    cdf = pd.concat([xdf,ydf])
    cdf["CID"] = i
    pdf = pd.concat([pdf,cdf])
    CIDs.append(i)
df = pd.DataFrame()
df["CID"] = CIDs
NN = NearestNeighbour(df,pdf,"x","y",ROIdic)
NN.show_np_phi()
#print(NN.D)
#plt.hist(NN.D[2],density=True)
#plt.plot(NN.np_p_of_d(NN.weights,2))

