# -*- coding: utf-8 -*-
"""
Created on Mon May  1 17:09:33 2023

@author: Raphael
"""

import roifile as rf
import numpy as np
from skimage.measure import grid_points_in_poly
import matplotlib.pyplot as plt
import time
import tifffile as tif
import os


path = "E:/Raphael/Data/3. Plasmid moves cargo into Lysosomes/1/FY_LAMP1/0.zip"

ROIs = rf.ImagejRoi.fromfile(path)
r = ([ROI for ROI in ROIs])[0]
coords = r.coordinates()
cs = np.subtract(coords,np.array([[r.left,r.top]]))
start_time = time.time()
image = grid_points_in_poly((r.right-r.left,r.bottom-r.top),cs)
print(time.time()-start_time)
plt.imshow(image)



path = "E:/Raphael/Data/test.zip"

ROI = rf.ImagejRoi.frompoints([[100, 100]],name="Hello")
ROI.tofile(path)


path = "E:/Raphael/Data/3. Plasmid moves cargo into Lysosomes/1/FY_LAMP1/0.tif"
with tif.TiffFile(path) as tiff:
    chdic = {}
    st = tiff.imagej_metadata["channel"]
    print(st)
    st2 = st.split(",")
    for st3 in st2:
        chdic[str(st3.split(":")[1].replace(" ","").replace("}",""))] = str(st3.split(":")[0].split("'")[1])
    print(chdic)
        
print(os.path.split(path))
        
        
        
a1 = np.array([[1,1],[1,1]])
a2 = np.array([[True],[False]])
print(a1*a2)
    
