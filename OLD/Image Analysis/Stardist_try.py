# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:39:34 2023

@author: Raphael
"""
def getimfolder(experiment, rep):
    import os
    basedir = ""
    rawdir = ""
    splitdir = ""
    data = "E:\Raphael\Data"
    for folder in os.listdir(data):
        if str(experiment) in folder:
            basedir = os.path.join(os.path.join(data, folder),str(rep))
    return basedir

import matplotlib.pyplot as plt
import os
import tifffile as tiff
experiment = 7
rep = 1
imageID = 2
basedir = getimfolder(experiment, rep)
splitdir = os.path.join(basedir,"Images_Split")
img = tiff.imread(os.path.join(splitdir, str(imageID)+".tif"))[0,:,:]
plt.imshow(img)

# Stardist segmentation

from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize

model = StarDist2D.from_pretrained('2D_versatile_fluo')

from skimage.transform import rescale, resize, downscale_local_mean
image_rescaled = rescale(img, 0.25, anti_aliasing=False)


labels, _ = model.predict_instances(normalize(image_rescaled))

plt.subplot(1,2,1)
plt.imshow(image_rescaled, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=image_rescaled))
plt.axis("off")
plt.title("prediction + input overlay")

# BF seg
from skimage.filters.rank import median
from skimage.morphology import disk, ball
from skimage import exposure
bfimg = tiff.imread(os.path.join(splitdir, str(imageID-1)+".tif"))[0,:,:]
plt.imshow(bfimg)
img_denoised = median(image, disk(5))
img_eq = exposure.equalize_hist(img_denoised)
bfimage_rescaled = rescale(img_eq, 4, anti_aliasing=False)
labels_rescaled = rescale(labels, 4, anti_aliasing=False)
plt.imshow(render_label(labels, img=bfimage_rescaled))
