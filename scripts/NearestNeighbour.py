# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 21:20:40 2023

@author: Raphael
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,cdist,squareform
from scipy.spatial import distance_matrix
from scipy.spatial import KDTree
import math
from shapely.geometry import Point, Polygon
import geopandas as gpd
from numpy.random import randint,normal
#from roifile import ImagejRoi

#from numpy.random import randint

def NearestNeighbour(X,Y,boundary,R,maxD=10,area=None):
	#In :
	#X: ndarray (nx,3) of coordinates of points in X (nx = number of points)
	#Y: ndarray (ny,3) of coordinates of points in Y (ny = number of points)
	#boundary: ndarray (nb,3) of coordinates of the cell boundary (nb = number of points)
	#R: ndarray (,nr) of radii to be checked (nr = number of radii)
	#area: area (float) of the cell
	#OUT:
	#AUC: Area under the curve of Ripley's L-function (float)
	#AUC_D: Area under the curve of Ripley's L-function without expected values (float)
    xtree = KDTree(X)
	#calculate distance between every point in X and Y
   	#calculate shortest distance between X and boundary
    D = xtree.query(Y)[0]
    Q = getQ(xtree,boundary)
   	#calculate boundary correction for every pair X and Y
    C = []
    Qr = []
   	# loop through all radii...
    for r in R:
   	#...and calculate k
       C.append(np.sum(D<r)/len(D))
       Qr.append(np.sum(Q<r)/len(Q))
    return np.array(C)-np.array(Qr)


def NN(D,C,r,area):
	#In :
	#D: ndarray (nx,ny) of distances between X and Y
	#C: ndarray (nx,ny) of boundary correction terms
	#r: radius to be measured (float)
	#area: area of the cell
	#OUT:
	#K: resulting k-value (float)
	#determine which points fall below radius
	I = D<r
	#calculate K
	K = np.sum(C,where=I)*area/D.size
	return K

def getQ(xtree, boundary):
	#In :
	#K: ndarray (,nr) of k-values for each radius
	#OUT:
	#L: resulting L-value (float)
    P = sample_points(boundary)
    Q = xtree.query(P)[0]
    return Q

def sample_points(boundary,number = 10000):
    gdf_poly = gpd.GeoDataFrame(index=["ROI"], geometry=[Polygon(boundary)])
    x = np.random.uniform(np.min(boundary.T[0]), np.max(boundary.T[0]), number )
    y = np.random.uniform(np.min(boundary.T[1]), np.max(boundary.T[1]), number )
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
