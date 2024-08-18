# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 17:49:54 2023

@author: Raphael
"""
import numpy as np

def manders (image1, image2, mask):
    image2[mask==False]=False
    m1 = np.sum(image1,where=image2)
    m2 = np.sum(image1,where=mask)
    return m1/m2

def adapted_manders (image1, image2, mask):
	m1 = np.mean(image1,where=image2)
	m2 = np.mean(image1,where=mask)
	return m1/m2

def manders_overlap (image1, image2, mask):
    ov = np.sum(np.logical_and(image1,image2),where=mask)
    m1 = np.sum(image1,where=mask)
    m2 = np.sum(image2,where=mask)
    return ov/m1, ov/m2