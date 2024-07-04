# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:08:38 2023

@author: Raphael
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,cdist,squareform
from scipy.spatial import distance_matrix
from scipy.spatial import KDTree
import math
#from roifile import ImagejRoi

#from numpy.random import randint

def ripley(X,Y,boundary,R,maxD=10,area=None):
	#In :
	#X: ndarray (nx,3) of coordinates of points in X (nx = number of points)
	#Y: ndarray (ny,3) of coordinates of points in Y (ny = number of points)
	#boundary: ndarray (nb,3) of coordinates of the cell boundary (nb = number of points)
	#R: ndarray (,nr) of radii to be checked (nr = number of radii)
	#area: area (float) of the cell
	#OUT:
	#AUC: Area under the curve of Ripley's L-function (float)
	#AUC_D: Area under the curve of Ripley's L-function without expected values (float)
    if area == None:
        area = getArea(boundary)
    xtree = KDTree(X)
    ytree = KDTree(Y)
    btree = KDTree(boundary)
	#calculate distance between every point in X and Y
    D = distance_matrix(X, Y).T
   	#calculate shortest distance between X and boundary
    E = btree.query(X)[0]
   	#calculate boundary correction for every pair X and Y
    C = correct_boundary(D,E)
    K = []
   	# loop through all radii...
    for r in R:
   	#...and calculate k
   		K.append(k_function(D,C,r,area))
   	#calculate L-function
    L = l_function(np.array(K))
   	#calculate AUC
    AUC = np.sum(L)
   	#correct AUC for assumed Poisson process
    AUC_D = np.sum(L-R)
    return L


def correct_boundary(D,E):
	#In :
	#D: ndarray (nx,ny) of distances between X and Y
	#E: ndarray (nx,ny) of distances between X and boundary
	#OUT:
	#C: ndarray (nx,ny) of boundary correction terms
	minD = np.minimum(D+0.0001,E)/(D+0.0001)
	C = 1/(1-(np.arccos(minD))/math.pi)
	return C

def k_function(D,C,r,area):
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

def l_function(K):
	#In :
	#K: ndarray (,nr) of k-values for each radius
	#OUT:
	#L: resulting L-value (float)
	L = np.sqrt(K/np.pi)
	return L

def getArea(boundary):
    x = boundary.T[0]
    y = boundary.T[1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
