# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:59:02 2023

@author: Raphael
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from roifile import ImagejRoi
import tifffile as tiff
from scipy import ndimage as ndi
from tqdm import tqdm
import time

from skimage.filters import gaussian,threshold_multiotsu,threshold_otsu
from skimage.restoration import rolling_ball
from skimage.segmentation import watershed,clear_border
from skimage.exposure import equalize_adapthist,rescale_intensity
from skimage.measure import regionprops_table, find_contours,blur_effect, points_in_poly,grid_points_in_poly,label,pearson_corr_coeff
from skimage.feature import peak_local_max

from scipy.stats import pearsonr


from findmaxima2d import find_maxima, find_local_maxima

basedir = "D:/Raphael/Data/3. Plasmid moves cargo into Lysosomes"
df = pd.read_excel(os.path.join(basedir,"Results.xlsx"))
ch_dic={"aSyn-GFP":0,"aSyn-AB":1,"LAMP1/LC3":2}
channel_aut = "LAMP1/LC3"
channel_aSyn = "aSyn-GFP"
channel_AB = "aSyn-AB"

def change_path(path, old="E", new="D"):
    return new+path[1::]

def getImage(path,ch,cn=None):
    #Implements opening of a specified image
    #Inputs:
        #path: path to the image
        #ch: channel name of the image
    #First read where the image is stored
    cID = cn
    with tiff.TiffFile(path) as image:
        #read image as dask array -° speeds up computation
        img = tiff.imread(path)
        if cn == None:
            #get the arrangement of desired channel
            C = image.imagej_metadata["channel"]
            I = C.split(",")
            cID = None
            for i in I:
                if ch in i:
                    cID = int(i.split(":")[1].replace(" ","").replace("}",""))
            #determine the number of the channel
        if len(img.shape)==4:
            img = img[:,cID,:,:]
            return np.array(img)
        else:
            img = img[cID,:,:]
            return np.array(img)
        #return as np array
        return np.array(img)

def getCellROI(CID):
    roi_path = change_path(df["ROI_path"][df["CID"]==CID].values[0])
    ROIs = ImagejRoi.fromfile(roi_path)
    for r in ROIs:
        if r.name == "Cell_"+str(CID):
            return r 

def crop(image,ROI):
    #Implements cropping of a 2d image
    #Inputs:
        #image: image to be cropped
        #ROI: list of coordinates specifying the cell boundaries
    #cropping of the image by reading out the bounding box information
    if len(image.shape)==3:
        cimage = image[:,int(ROI.top):int(ROI.bottom),int(ROI.left):int(ROI.right)]
        return cimage
    if len(image.shape)==4:
        cimage = image[:,:,int(ROI.top):int(ROI.bottom),int(ROI.left):int(ROI.right)]
        return cimage
    else:
        cimage = image[int(ROI.top):int(ROI.bottom),int(ROI.left):int(ROI.right)]
        return cimage


def denoise_p(CID,sigma1=5,radius=15,sigma2=5):
    
    path = change_path(df["path"][df["CID"]==CID].values[0])
    ROI = getCellROI(CID)
    img = tiff.imread(path)
    # Images are cropped
    cimg = crop(img,ROI)
    #CHeck if image is 3d
    
        #Since each slice of a 3d image gets processed individually the slices have to be temporarily stored in an array and then put together afterwards
    newimg = []
    #Loop through all slices
    for st in range(img.shape[0]):
        stimg = []
        for ch in range(img.shape[1]):
            #Apply first gaussian smoothing
            #Low sigma leads to little change in detail but removal of fine high frequency noise
            gimg = gaussian(cimg[st,ch,:,:],sigma1)
            #Perform rolling ball background estimation; Low radius will remove large amounts of cytoplasmic detail but keep the peaks intact
            bg = rolling_ball(gimg, radius=radius)
            #Gaussian smoothing of the background
            bg = gaussian(bg,sigma2)
            #Subtract background
            bgsub = gimg-bg
            #Store the denoised slice in the prevously created array
            stimg.append(bgsub)
        newimg.append(np.array(stimg))
    #return the slices after putting them back together
    return np.array(newimg)

def denoise(CID,ch,sigma1=0.2,radius=15,sigma2=2):
    
    path = change_path(df["path"][df["CID"]==CID].values[0])
    ROI = getCellROI(CID)
    img = getImage(path,ch,cn=ch_dic[ch])
    # Images are cropped
    cimg = crop(img,ROI)
    #CHeck if image is 3d
    if len(img.shape)==3:
        #Since each slice of a 3d image gets processed individually the slices have to be temporarily stored in an array and then put together afterwards
        newimg = []
        #Loop through all slices
        for st in range(img.shape[0]):
            #Apply first gaussian smoothing
            #Low sigma leads to little change in detail but removal of fine high frequency noise
            gimg = gaussian(cimg[st,:,:],sigma1)
            #Perform rolling ball background estimation; Low radius will remove large amounts of cytoplasmic detail but keep the peaks intact
            bg = rolling_ball(gimg, radius=radius)
            #Gaussian smoothing of the background
            bg = gaussian(bg,sigma2)
            #Subtract background
            bgsub = gimg-bg
            #Store the denoised slice in the prevously created array
            newimg.append(bgsub)
        #return the slices after putting them back together
        return np.array(newimg)
    


 
def extract_points(CID,channel_aut="LAMP1/LC3",channel_aSyn="aSyn-GFP",channel_AB="aSyn-AB"):
    ROI = getCellROI(CID)
    aut_img = rescale_intensity(denoise(CID,channel_aut),out_range="uint8")
    aSyn_img = rescale_intensity(denoise(CID,channel_aSyn),out_range="uint8")
    AB_img = rescale_intensity(denoise(CID,channel_AB),out_range="uint8")
    gridmask = grid_points_in_poly(aut_img[0,:,:].shape,ROI.coordinates()-[[ROI.left,ROI.top]])
    pdf = pd.DataFrame()
    Aarea = 0
    for st in tqdm(range(aut_img.shape[0]),desc=str(CID)):
        pimp = aut_img[st,:,:]
        local_max = find_local_maxima(pimp)
        y, x, regs = find_maxima(pimp,local_max,10)
        coo = np.array([y,x])
        #coo= peak_local_max(pimp,threshold_abs=t1,min_distance=20).T
        binary = np.zeros(pimp.shape, dtype=bool)
        if len(np.unique(pimp)) > 2:
            binary = (pimp > threshold_multiotsu(pimp)[0])
        mask = np.zeros(binary.shape, dtype=bool)
        mask[tuple(coo)] = True
        mask[binary == False] = False
        markers, _ = ndi.label(mask)
        labels = watershed(-aut_img[st,:,:], markers, mask=binary)
        labels[gridmask == False] = 0
        #contours = find_contours(labels>0)
        rp = pd.DataFrame(regionprops_table(labels,properties=('label',"area")))
        Aarea = Aarea + rp["area"].sum()
        data = {}
        data["x"] = coo[1]
        data["y"] = coo[0]
        data["z"] = [st]*coo[1].shape[0]
        data["InCytoplasm"] = mask[tuple(coo)]
        data["InAutophagosome"] = [True]*coo[1].shape[0]
        data["Channel"] = [channel_aut]*coo[1].shape[0]
        data["CID"] = [CID]*coo[1].shape[0]
        pdf = pd.concat([pdf,pd.DataFrame(data=data)])
        
        
        
        
        
        pimp = aSyn_img[st,:,:]
        #pimp[label==False] = 0
        local_max = find_local_maxima(pimp)
        y, x, regs = find_maxima(pimp,local_max,10)
        coo = np.array([y,x])
        labs = labels[tuple(coo)]
        pol = points_in_poly(coo.T,ROI.coordinates()-[[ROI.left,ROI.top]])
        #plt.plot(ROI.coordinates().T[0]-ROI.left,ROI.coordinates().T[1]-ROI.top)
        data = {}
        data["x"] = coo[1]
        data["y"] = coo[0]
        data["z"] = [st]*coo[1].shape[0]
        data["InCytoplasm"] = pol
        data["InAutophagosome"] = labs>0
        data["Channel"] = [channel_aSyn]*coo[1].shape[0]
        data["CID"] = [CID]*coo[1].shape[0]
        pdf = pd.concat([pdf,pd.DataFrame(data=data)])
        
        
        #path = df["path"][df["CID"]==CID].values[0]
        #testimg = crop(getImage(path,"",cn=0),ROI)
        #plt.imshow(testimg[st,:,:])
        #plt.plot(coo[1],coo[0],".r")
        
        
        
        pimp = AB_img[st,:,:]
        #pimp[label==False] = 0
        local_max = find_local_maxima(pimp)
        y, x, regs = find_maxima(pimp,local_max,10)
        coo = np.array([y,x])
        labs = labels[tuple(coo)]
        pol = points_in_poly(coo.T,ROI.coordinates()-[[ROI.left,ROI.top]])
        #plt.plot(ROI.coordinates().T[0]-ROI.left,ROI.coordinates().T[1]-ROI.top)
        data = {}
        data["x"] = coo[1]
        data["y"] = coo[0]
        data["z"] = [st]*coo[1].shape[0]
        data["InCytoplasm"] = pol
        data["InAutophagosome"] = labs>0
        data["Channel"] = [channel_AB]*coo[1].shape[0]
        data["CID"] = [CID]*coo[1].shape[0]
        pdf = pd.concat([pdf,pd.DataFrame(data=data)])
        
    df.loc[df.CID == CID, "Autophagosome_area"] = Aarea     
        
        
    return pdf

def calc_area(img,timg):
    bimg = img > threshold_otsu(timg)
    return bimg

def extract_manders(CID,channel_aut="LAMP1/LC3",channel_aSyn="aSyn-GFP",channel_AB="aSyn-AB"):
    denoised_path = df["denoised_path"][df["CID"]==CID].values[0]
    ROI = getCellROI(CID)
    aut_img = rescale_intensity(getImage(denoised_path,"",cn=ch_dic[channel_aut]),out_range="uint8")
    aSyn_img = rescale_intensity(getImage(denoised_path,"",cn=ch_dic[channel_aSyn]),out_range="uint8")
    AB_img = rescale_intensity(getImage(denoised_path,"",cn=ch_dic[channel_AB]),out_range="uint8")
    tsize = aut_img.shape[0]*aut_img.shape[1]*aut_img.shape[2]
    gridmask = grid_points_in_poly(aut_img[0,:,:].shape,ROI.coordinates()-[[ROI.left,ROI.top]])
    data = {"Seg_GFP":[],"Seg_AB":[],"Seg_LC3":[],"O_GFP+LC3":[],"O_GFP+AB":[],"O_AB+LC3":[],"Int_GFP":[],"Int_AB":[],"Int_LC3":[],"Int_GFP+LC3":[],"Int_LC3+AB":[],"Int_LC3+GFP":[],"Int_AB+LC3":[]}
    for st in range(aut_img.shape[0]):
        lc3 = calc_area(aut_img[st,:,:],aut_img)
        GFP = calc_area(aSyn_img[st,:,:],aSyn_img)
        AB = calc_area(AB_img[st,:,:],AB_img)
        GFP_lc3 = np.logical_and(GFP,lc3)
        GFP_AB = np.logical_and(GFP,AB)
        AB_lc3 = np.logical_and(AB,lc3)
        data["Seg_GFP"].append(np.sum(np.logical_and(GFP,gridmask))/tsize)
        data["Seg_AB"].append(np.sum(np.logical_and(AB,gridmask))/tsize)
        data["Seg_LC3"].append(np.sum(np.logical_and(lc3,gridmask))/tsize)
        data["O_GFP+LC3"].append(np.sum(np.logical_and(GFP_lc3,gridmask))/tsize)
        data["O_GFP+AB"].append(np.sum(np.logical_and(GFP_AB,gridmask))/tsize)
        data["O_AB+LC3"].append(np.sum(np.logical_and(AB_lc3,gridmask))/tsize)
        I_lc3 = aut_img[st,:,:]
        I_lc3[gridmask == False] = 0
        I_GFP = aSyn_img[st,:,:]
        I_GFP[gridmask == False] = 0
        I_AB = AB_img[st,:,:]
        I_AB[gridmask == False] = 0
        data["Int_GFP"].append(np.sum(I_GFP))
        data["Int_AB"].append(np.sum(I_AB))
        data["Int_LC3"].append(np.sum(I_lc3))
        data["Int_GFP+LC3"].append(np.sum(I_GFP*lc3))
        data["Int_AB+LC3"].append(np.sum(I_AB*lc3))
        data["Int_LC3+GFP"].append(np.sum(I_lc3*GFP))
        data["Int_LC3+AB"].append(np.sum(I_lc3*AB))
    pdf = pd.DataFrame(data=data).sum().to_frame().transpose()
    return pdf


#extract_points(19)

def extract_all_manders(minPID = 0,df=df):
    pdf = pd.DataFrame()
    for PID in tqdm(df["PID"].unique()):
        if PID < minPID:
            continue
        for CID in df["CID"][df["PID"]==PID].unique():
            spdf = extract_manders(CID)
            spdf["CID"] = CID
            pdf = pd.concat([pdf,spdf])
    df = df.merge(pdf,on="CID")
    df.to_excel(os.path.join(basedir,"Results.xlsx"), index=False)
    return pdf

#test = extract_manders(19)

def extract_all(minPID = 0):

    for PID in tqdm(df["PID"].unique()):
        if PID < minPID:
            continue
        pdf = pd.DataFrame()
        for CID in df["CID"][df["PID"]==PID].unique():
            pdf = pd.concat([pdf,extract_points(CID)])
        pdf.to_csv(df["coord_path"][df["PID"]==PID].values[0], sep="\t")
    
    df.to_excel(os.path.join(basedir,"Results.xlsx"), index=False)


def denoise_all():
    denpath = []
    for CID in tqdm(df["CID"].unique()):
        denoised_path = os.path.join(os.path.join(os.path.join(basedir,str(df["Replicate"][df["CID"]==CID].values[0])),str(df["Group1"][df["CID"]==CID].values[0])+"_"+str(df["Group2"][df["CID"]==CID].values[0])),str(CID)+"_denoised.tif")
        den_img = rescale_intensity(denoise_p(CID),out_range="uint8")
        tiff.imwrite(denoised_path, den_img, imagej=True)
        img = tiff.imread(denoised_path)
        denpath.append(denoised_path)
    df["denoised_path"] = denpath
    df.to_excel(os.path.join(basedir,"Results.xlsx"), index=False)
    return

def extract_pearson(CID,channel_aut="LAMP1/LC3",channel_aSyn="aSyn-GFP",channel_AB="aSyn-AB"):
    denoised_path = df["denoised_path"][df["CID"]==CID].values[0]
    ROI = getCellROI(CID)
    aut_img = rescale_intensity(getImage(denoised_path,"",cn=ch_dic[channel_aut]),out_range="uint8")
    aSyn_img = rescale_intensity(getImage(denoised_path,"",cn=ch_dic[channel_aSyn]),out_range="uint8")
    AB_img = rescale_intensity(getImage(denoised_path,"",cn=ch_dic[channel_AB]),out_range="uint8")
    gridmask = grid_points_in_poly(aut_img[0,:,:].shape,ROI.coordinates()-[[ROI.left,ROI.top]])
    GFParr = []
    ABarr = []
    LC3arr = []
    for st in range(aut_img.shape[0]):
        data = pd.DataFrame()
        data["GFP"] = aSyn_img[st,:,:].ravel()
        data["AB"] = AB_img[st,:,:].ravel()
        data["LC3"] = aut_img[st,:,:].ravel()
        data["mask"] = gridmask.ravel()
        data = data[data["mask"]==True]
        GFParr.append(data["GFP"].values)
        ABarr.append(data["AB"].values)
        LC3arr.append(data["LC3"].values)
    GFParr = np.concatenate(list(GFParr),axis=None)
    ABarr = np.concatenate(list(ABarr),axis=None)
    LC3arr = np.concatenate(list(LC3arr),axis=None)
    GFP_pears_s,GFP_pears_p = pearsonr(GFParr,LC3arr)
    AB_pears_s,AB_pears_p = pearsonr(ABarr,LC3arr)
    return pd.DataFrame(data={"GFP_pears_s":GFP_pears_s,"GFP_pears_p":GFP_pears_p,"AB_pears_s":AB_pears_s,"AB_pears_p":AB_pears_p}, index=[CID])


def extract_all_pearson(minPID = 0,df=df):
    pdf = pd.DataFrame()
    for PID in tqdm(df["PID"].unique()):
        if PID < minPID:
            continue
        for CID in df["CID"][df["PID"]==PID].unique():
            spdf = extract_pearson(CID)
            spdf["CID"] = CID
            pdf = pd.concat([pdf,spdf])
    df = df.merge(pdf,on="CID")
    df.to_excel(os.path.join(basedir,"Results.xlsx"), index=False)
    return pdf

denoise_all()

#pdf = extract_all_manders()
#df = df.merge(pdf,on="CID")
#df.to_excel(os.path.join(basedir,"Results.xlsx"), index=False)

#plt.imshow(aut_img[9,:,:]> threshold_otsu(aut_img[9,:,:]))
#pdf = extract_all_pearson()