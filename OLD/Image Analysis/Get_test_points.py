# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:19:16 2023

@author: Raphael
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import pickle
import Opening
import os

experiment = 3
basedir = Opening.getimfolder(experiment)
print(basedir)

num = 50
dic = {}


for cID in range(100):
    print(cID)
    dic[cID] = {}
    P = np.zeros((250,250,250))
    indL = np.reshape(np.indices(P.shape).T,(1,1,-1,3))[0][0]
    LAMPI = np.random.choice(np.arange(0,indL.shape[0]),100)
    LAMP = indL[LAMPI]
    indQ = np.reshape(np.indices(P.shape).T,(1,1,-1,3))[0][0]
    QI = np.random.choice(np.arange(0,indQ.shape[0]),100)
    Q = indQ[QI]
    Ptemp = P.flatten()
    Ptemp[LAMPI] = 1
    P = Ptemp.reshape(P.shape)
    dic[cID]["LAMP"] = LAMP
    dic[cID]["aSyn"] = {}
    dic[cID]["Q"] = Q
    print(np.unique(P, return_counts=True))
    S = [0,0.1,0.2,0.5,0.7,1,2,3,4,5,6,7,8,9,10]
    for sigma in S:
        pmap = gaussian_filter(P,sigma,mode="wrap").reshape(1,1,-1)[0][0]
        Z = np.sum(P)
        pmap = pmap/Z
        ind = np.reshape(np.indices(P.shape).T,(1,1,-1,3))[0][0]
        aSynI = np.random.choice(np.arange(0,ind.shape[0]),100,p=pmap)
        aSyn = ind[aSynI]
        dic[cID]["aSyn"][sigma] = aSyn

with open(os.path.join(basedir,'coords.pickle'), 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
