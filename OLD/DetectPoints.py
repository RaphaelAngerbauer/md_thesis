# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 15:13:51 2023

@author: Raphael
"""
import numpy as np
from skimage.measure import points_in_poly
from skimage.morphology import extrema

def detect_points(image,ROI,ntol=10):
    ROIcoords = ROI.coordinates()-[[ROI.left,ROI.top]]
    local_max = find_local_maxima(image)
    y, x, regs = find_maxima(image,local_max,ntol)
    points = np.array([y,x]).T
    inCell = points_in_poly(points,ROIcoords)
    return points[inCell]