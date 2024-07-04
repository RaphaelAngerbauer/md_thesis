# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:48:20 2023

@author: Raphael
"""

import pandas as pd
import os
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import stats
from scipy import ndimage as ndi
from skimage.util import img_as_float
from skimage.feature import peak_local_max
from skimage import filters

from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize

from skimage import morphology
from skimage.filters import threshold_multiotsu, threshold_otsu, sobel
from skimage.segmentation import watershed, expand_labels
from skimage.filters import rank, laplace
from skimage.measure import regionprops
from skimage import measure
from skimage import restoration




def getimfolder(experiment):
    import os
    basedir = ""
    data = "E:\Raphael\Data"
    for folder in os.listdir(data):
        if str(experiment) in folder:
            basedir = os.path.join(data, folder)
    return basedir

def readimages(rawdir, splitdir, ID, NDID):
    import os
    import tifffile as tiff
    import nd2
    data = {"ID":[], "PID":[], "Stack":[], "Channel": [], "xlen": [], "ylen":[], "zlen": [], "Group1":[], "Group2":[]}
    for ndimg in os.listdir(rawdir):
        if "Deconvolved" in ndimg:
            PID = 0
            iml = os.path.join(rawdir, ndimg)
            with nd2.ND2File(iml) as ndfile:
                meta = ndfile.metadata
                shape = ndfile.shape
                for series in range(shape[0]):
                    stackID = 0
                    for stack in range(shape[1]):
                        for i in range(len(meta.channels)):
                            img = ndfile.asarray(PID)
                            img = img[:,stackID,i,:,:]
                            tiff.imwrite(os.path.join(splitdir,str(ID))+".tif", img)
                            chan = meta.channels[i].channel.name
                            xlen, ylen, zlen = meta.channels[i].volume.axesCalibration
                            data["ID"].append(str(ID))
                            data["PID"].append(str(PID))
                            data["NDID"].append(NDID)
                            data["Stack"].append(str(stackID))
                            data["Channel"].append(str(chan))
                            data["xlen"].append(xlen)
                            data["ylen"].append(ylen)
                            data["zlen"].append(zlen)
                            data["Group1"].append(ndimg.split("_")[0])
                            data["Group2"].append(ndimg.split("_")[1].split(" ")[0])
                            print(ID)
                            ID = ID+1
                        NDID+=1
                        stackID = stackID+1
                    PID = PID+1
    return data

def splitimages(experiment, rep):
    import os
    import pandas as pd
    basedir = getimfolder(experiment)
    rawdir = os.path.join(basedir,str(rep))
    splitdir = os.path.join(basedir,"Images")
    info = os.path.join(basedir, "Info.xlsx")
    if os.path.exists(info) == True:
        old_df = pd.read_excel(info)
        ID = max(old_df["ID"])
        NDID = max(old_df["NDID"])
        data = readimages(rawdir, splitdir, ID, NDID)
        new_df = pd.DataFrame(data = data)
        df = pd.concat(old_df,new_df)
    else:
        ID = 0
        data = readimages(rawdir, splitdir, ID)
        df = pd.DataFrame(data = data)
    df.to_excel(info,index=False)
    

    
def rescale_before(img, pixlenimg):
    from skimage.transform import rescale
    pixlensd = 0.3488
    scaling_factor = pixlenimg/pixlensd
    image_rescaled = rescale(img, scaling_factor, anti_aliasing=True)
    return image_rescaled

def rescale_after(img, dims):
    from skimage.transform import resize
    image_resized = resize(img, dims, preserve_range=True, order=0, anti_aliasing=False).astype('uint8')
    return image_resized


def createkernel(kernel, level):
    zeroc = (2**(level-1))-1
    nklen = kernel.shape[1] + (kernel.shape[1]-1)*zeroc
    new_kernel = np.zeros((1, nklen), dtype = kernel.dtype)
    for i in range(kernel.shape[1]):
        index = i + i*zeroc
        print(index)
        new_kernel[0][index] = kernel[0][i]
    return new_kernel

def atrousconvolution(image, P_old, level , og_kernel):
    kernel = createkernel(og_kernel, level)
    Ai = signal.convolve2d(image, kernel, boundary = "symm", mode = "same")
    A = signal.convolve2d(Ai, kernel.T, boundary = "symm", mode = "same")
    W = image - A
    t = 3*(stats.median_abs_deviation(W)/0.67)
    T = W
    T[T<t] = 0
    P = T*P_old
    return P, W

def wavelet_detection(image, og_kernel, max_level):
    level = 1
    P = np.ones(image.shape)
    cimage = image
    while level < max_level+1:
        cimage, P = atrousconvolution(image, P, level , og_kernel)
        level = level + 1
    threshold = filters.threshold_otsu(P)
    coordinates = peak_local_max(P, min_distance=5, threshold_abs = threshold)
    return coordinates, P

def getID(df, PID, stack, Group1, Group2, Channel):
    ID = []
    if Channel != None:
        ID = df["ID"][(df["PID"]==PID) & (df["Stack"]==stack) &( df["Group1"]==Group1) & (df["Group2"]==Group2) & (df["Channel"] == Channel)].values
    else:
        ID = df["ID"][(df["PID"]==PID) & (df["Stack"]==stack) &( df["Group1"]==Group1) & (df["Group2"]==Group2)].values
    return ID

def getvarLaplace(img, labeled_img, label):
    binary = labeled_img == label
    VarLap = laplace(img, mask = binary).var()
    return VarLap

def subsegment():
    
    return

def segment_cells(df, PID, NDID, stack, Group1, Group2, model, basedir):
    if model == None:
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    imageID = int(getID(df, PID, stack, Group1, Group2, "DAPI"))
    image = img_as_float(tiff.imread(os.path.join(imagedir, str(imageID)+".tif"))[0,:,:])
    pixlenimg = df["xlen"][df["ID"]==imageID]
    img_rescaled = rescale_before(image, pixlenimg)
    labelssd, _ = model.predict_instances(normalize(img_rescaled))
    labels_rescaled = rescale_after(labelssd, image.shape)
    ID = int(getID(df, PID, stack, Group1, Group2, "GFP"))
    img = img_as_float(tiff.imread(os.path.join(imagedir, str(ID)+".tif"))[0,:,:])
    thresholds = threshold_multiotsu(img)
    thresholds[::-1].sort()
    labeled = labels_rescaled
    for t in thresholds:
        binary = img >= t
        binary[labeled != 0] = True
        labeled = watershed(-img, markers = labeled, mask = binary)
    tempprops =  measure.regionprops_table(labeled, img, properties=['label', 'bbox','intensity_mean'])
    finallabel = np.zeros(labels_rescaled.shape)
    propdf = pd.DataFrame(tempprops)
    propdf = propdf.sort_values(by=['intensity_mean'], ascending=False)
    bbox0 = propdf.loc[:,"bbox-0"].values
    bbox1 = propdf.loc[:,"bbox-1"].values
    bbox2 = propdf.loc[:,"bbox-2"].values
    bbox3 = propdf.loc[:,"bbox-3"].values
    LLabel = propdf.loc[:,"label"].values
    for i in range(len(LLabel)):
        l = LLabel[i]
        mask_crop = labeled[bbox0[i]:bbox2[i],bbox1[i]:bbox3[i]]
        crop = img[bbox0[i]:bbox2[i],bbox1[i]:bbox3[i]]
        t = threshold_multiotsu(crop)
        labels = np.zeros(mask_crop.shape)
        labels[mask_crop == l] = 0
        labels[crop > t[0]] = 2
        labels[mask_crop != l] = 1
        elevation_map = sobel(crop)
        seg = watershed(elevation_map, labels)
        seg = seg-1
        segup = np.zeros(finallabel.shape)
        segup[bbox0[i]:bbox2[i],bbox1[i]:bbox3[i]] = seg
        segup[finallabel > 0] = 0
        finallabel[segup > 0] = segup
    plt.imshow(finallabel)
    tiff.imwrite(os.path.join(os.path.join(basedir,"Masks_Nuclei"),str(NDID))+".tif", labels_rescaled)
    tiff.imwrite(os.path.join(os.path.join(basedir,"Masks_Cells"),str(NDID))+".tif", finallabel)
    centroidsN = {}
    AreaN = {}
    LaplaceN = {}
    props = regionprops(label_image = labels_rescaled, intensity_image = image)
    for p in props:
        centroid = p.centroid
        centroid = tuple(map(int, centroid))
        label = p.label
        centroidsN[str(label)] = centroid
        AreaN[str(label)] = p.area
        VarLap = getvarLaplace(image, labeled, label)
        LaplaceN[str(label)] = VarLap 
    centroidsC = {}
    GFP = {}
    LaplaceC = {}
    AreaC = {}
    bb = {}
    props = regionprops(label_image = finallabel, intensity_image = img)
    for p in props:
        centroid = p.centroid
        centroid = tuple(map(int, centroid))
        label = p.label
        centroidsC[str(label)] = centroid
        GFP[str(label)] = p.intensity_mean
        AreaC[str(label)] = p.area
        VarLap = getvarLaplace(image, labeled, label)
        LaplaceC[str(label)] = VarLap 
        bb[str(label)] = p.bbox
    cinfo = os.path.join(basedir, "CellInfo.xlsx")
    data = {"CID":[], "NDID":[], "Label":[], "CentroidNx":[], "CentroidNy":[], "CentroidCx":[], "CentroidCy":[], "GFP":[],"AreaN":[],"AreaC":[],"LaplaceN":[],"LaplaceC":[],"min_row":[],"min_col":[],"max_row":[],"max_col":[],"Include":[]}
    CID = 0
    for i in np.unique(labeled):
        if i == 0:
            continue
        else:
            data["CID"].append(CID)
            data["NDID"].append(NDID)
            data["Label"].append(i)
            data["CentroidNx"].append(centroidsN[str(i)][0])
            data["CentroidNy"].append(centroidsN[str(i)][1])
            data["CentroidCx"].append(centroidsC[str(i)][0])
            data["CentroidCy"].append(centroidsN[str(i)][1])
            data["GFP"].append(GFP[str(i)])
            data["AreaN"].append(AreaN[str(i)])
            data["AreaC"].append(AreaC[str(i)])
            data["LaplaceN"].append(LaplaceN[str(i)])
            data["LaplaceC"].append(LaplaceC[str(i)])
            data["min_row"].append(bb[str(i)][0])
            data["min_col"].append(bb[str(i)][1])
            data["max_row"].append(bb[str(i)][2])
            data["max_col"].append(bb[str(i)][3])
            data["Include"].append(True)
            CID+=1
    if os.path.exists(cinfo) == True:
        old_df = pd.read_excel(cinfo)
        oldID = max(old_df["CID"])
        new_df = pd.DataFrame(data = data)
        new_df["CID"] = new_df["CID"]+oldID+1
        cdf = pd.concat([old_df,new_df])
        cdf.to_excel(cinfo,index=False)
    else:
        cdf = pd.DataFrame(data = data)
        cdf.to_excel(cinfo,index=False)
    return model

def extractpuncta(image, CID, cinfo, kernel, max_level, stain):
    r = cinfo[cinfo["CID"]==CID]
    crop = image[int(r["min_row"]):int(r["max_row"]),int(r["min_col"]):int(r["max_col"])]
    maskc = tiff.imread(os.path.join(maskdir, str(int(r["NDID"]))+".tif"))
    mask_crop = maskc[int(r["min_row"]):int(r["max_row"]),int(r["min_col"]):int(r["max_col"])]
    t = threshold_multiotsu(crop)
    labels = mask_crop
    labels[mask_crop != int(r["Label"])] = 1
    labels[mask_crop == int(r["Label"])] = 0
    labels[crop > t[0]] = 2
    elevation_map = sobel(crop)
    seg = watershed(elevation_map, labels)
    coordinates, P = wavelet_detection(crop, kernel, max_level)
    binary = P >= threshold_multiotsu(P)[-1]
    binary[seg != 2] = 0
    templabel = measure.label(binary)
    exlabels = expand_labels(templabel, distance = 10)
    lbinary = exlabels
    lbinary[seg == 2] = 0
    #print(np.unique(lbinary))
    #plt.imshow(lbinary)
    for remove in np.unique(lbinary):
        if remove == 0:
            continue
        else:
            templabel[templabel == remove] = 0
            #continue
    binary[templabel == 0] = 0
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)))
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary)
    
    plt.imshow(labels)
    tiff.imwrite(os.path.join(os.path.join(basedir,"Masks_Puncta"),str(CID))+"_"+stain+".tif", labels)
    pinfo = os.path.join(basedir, "PunctaInfo.xlsx")

    data = {"Puncta_ID":[], "CID":[], "Label":[], "Stain":[], "Centroidx":[], "Centroidy":[], "Intensity":[],"Area":[],"Laplace":[],"min_row":[],"min_col":[],"max_row":[],"max_col":[],"Include":[]}
    centroids = {}
    intensity = {}
    Laplace = {}
    Area = {}
    bb = {}
    props = regionprops(label_image = labels, intensity_image = crop)
    for p in props:
        centroid = p.centroid
        centroid = tuple(map(int, centroid))
        label = p.label
        centroids[str(label)] = centroid
        intensity[str(label)] = p.intensity_mean
        Area[str(label)] = p.area
        VarLap = getvarLaplace(crop, mask_crop, label)
        Laplace[str(label)] = VarLap 
        bb[str(label)] = p.bbox
    Puncta_ID = 0
    for i in np.unique(labels):
        if i == 0:
            continue
        else:
            data["Puncta_ID"].append(Puncta_ID)
            data["CID"].append(CID)
            data["Label"].append(i)
            data["Stain"].append(stain)
            data["Centroidx"].append(centroids[str(i)][0])
            data["Centroidy"].append(centroids[str(i)][1])
            data["Intensity"].append(intensity[str(i)])
            data["Area"].append(Area[str(i)])
            data["Laplace"].append(Laplace[str(i)])
            data["min_row"].append(bb[str(i)][0])
            data["min_col"].append(bb[str(i)][1])
            data["max_row"].append(bb[str(i)][2])
            data["max_col"].append(bb[str(i)][3])
            data["Include"].append(True)
            Puncta_ID+=1
    if os.path.exists(pinfo) == True:
        old_df = pd.read_excel(os.path.join(basedir, "PunctaInfo.xlsx"))
        oldID = max(old_df["Puncta_ID"])
        new_df = pd.DataFrame(data = data)
        new_df["Puncta_ID"] = new_df["Puncta_ID"]+oldID+1
        cdf = pd.concat([old_df,new_df])
        cdf.to_excel(pinfo,index=False)
    else:
        cdf = pd.DataFrame(data = data)
        cdf.to_excel(pinfo,index=False)  
    return

max_level = 4
experiment = 3
rep = 1
PID = 0
NDID = 0
stack = 0
Group1 = "FY"
Group2 = "LAMP1"
Channel1 = "DAPI"
Channel2 = "GFP"
Channel3 = "568_2"

kernel = np.array([[1/16, 1/4, 3/8, 1/4, 1/16]])
#splitimages(experiment, rep)


basedir = getimfolder(experiment)
imagedir = os.path.join(basedir, "Images")
maskdir = os.path.join(basedir, "Masks_Cells")
info = os.path.join(basedir, "Info.xlsx")
df = pd.read_excel(info)
imageID = int(getID(df, PID, stack, Group1, Group2, Channel3))
image = img_as_float(tiff.imread(os.path.join(imagedir, str(imageID)+".tif"))[0,:,:])

        
#Segmentation
model = None
model = segment_cells(df, PID, NDID, stack, Group1, Group2, model, basedir)

# Puncta
cinfo = pd.read_excel(os.path.join(basedir, "CellInfo.xlsx"))
CID = 26
stain1 = "LAMP1"
imageID3 = int(getID(df, PID, stack, Group1, Group2, Channel3))
image3 = img_as_float(tiff.imread(os.path.join(imagedir, str(imageID3)+".tif"))[0,:,:])
plt.imshow(image3)
extractpuncta(image3, CID, cinfo, kernel, max_level, stain1)
stain2 = "GFP"
imageID2 = int(getID(df, PID, stack, Group1, Group2, Channel2))
image2 = img_as_float(tiff.imread(os.path.join(imagedir, str(imageID2)+".tif"))[0,:,:])
plt.imshow(image2)
extractpuncta(image2, CID, cinfo, kernel, max_level, stain2)

r = cinfo[cinfo["CID"]==CID]
crop = image2[int(r["min_row"]):int(r["max_row"]),int(r["min_col"]):int(r["max_col"])]
plt.imshow(crop)

maskc = tiff.imread(os.path.join(maskdir, str(int(r["NDID"]))+".tif"))
mask_crop = maskc[int(r["min_row"]):int(r["max_row"]),int(r["min_col"]):int(r["max_col"])]

plt.imshow(mask_crop)



#Subsegment
LLabel = 7
imageID2 = int(getID(df, PID, stack, Group1, Group2, "GFP"))
image2 = img_as_float(tiff.imread(os.path.join(imagedir, str(imageID2)+".tif"))[0,:,:])
maskc = tiff.imread(os.path.join(maskdir, str(int(NDID))+".tif"))
props = regionprops(label_image = maskc)
for p in props:
    if p.label == LLabel:
        


labelimg = 
sublabeled = subsegment()
