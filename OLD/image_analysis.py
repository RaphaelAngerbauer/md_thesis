# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:06:31 2023

@author: Raphael
"""

from skimage.restoration import rolling_ball
from skimage.feature import peak_local_max
from skimage.filters import gaussian,threshold_multiotsu,threshold_otsu
from skimage.transform import resize,rescale
from skimage.segmentation import watershed,clear_border
from skimage.measure import find_contours,regionprops_table,grid_points_in_poly,points_in_poly
from skimage.exposure import equalize_adapthist,rescale_intensity
from skimage import morphology

from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize

from scipy import ndimage as ndi

import pandas as pd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time

import nd2
import tifffile as tiff
from tqdm import tqdm
import json

from roifile import ImagejRoi,roiread,roiwrite
from findmaxima2d import find_maxima, find_local_maxima

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
            
def crop(image,ROI):
    #Implements cropping of a 2d image
    #Inputs:
        #image: image to be cropped
        #ROI: list of coordinates specifying the cell boundaries
    #cropping of the image by reading out the bounding box information
    if len(image.shape)==3:
        cimage = image[:,int(ROI.top):int(ROI.bottom),int(ROI.left):int(ROI.right)]
        return cimage
    else:
        cimage = image[int(ROI.top):int(ROI.bottom),int(ROI.left):int(ROI.right)]
        return cimage

def intensity_median(label, intensity):
    return np.median(intensity[label])


class registration(object):
    #The first class of functions splits up the hard to edit stacked ".nd" files into ImageJ TIFF files.
    #Usually when using the multipoint aquisition feature images are stored in a series. This means that usually one stores one image per condition
    #Drawing ROIs on these is dificult hence the need for splitting them up.
    #The resulting TIFF files are stored in a new Folder inside of the original directory, that is named after the original file
    #This means that if the original file is named "BL_BL.nd2" the resulting TIFF files are stored inside the folder "BL_BL"
    #The images are assinged an ID and named as such; so the name of a resulting TIFF could be 14.tif with 14 being the ID
    #The function requires a basedir, meaning the directory where all the experiment data is stored. There are 3 options how this can be specified:
        #1. basedir is given directly as a path
        #2. the ID of the experiment if the data folder is in the current working directory
        #3. The ID of the experiment + the folder of all the data
    #Inputs:
        #exID: (optional) int or str; Identifier of the experiment; needs to be the first character of the folder where the Experiment is stored; if not provided, basedir needs to be provided
        #basedir: (optional) str; Folder location of the experiment data; Either "basedir" or "data" need to be specified if the Data is not stored in the same directory as the current working directory
        #data: (optional) str; Folder location of all data; Folder location of the experiment data; Either "basedir" or "data" need to be specified if the Data is not stored in the same directory as the current working directory
        #Deconvolution: (optional) bool; Specify "True" if Deconvolution was used before to only split these images
        #maxPID: (optional) int; Specify the maximum PID previously used
        #reprange: (optional) int: Specify the amount of replicates in the experiment; needs to be done oly when there are more than 10
        #replicateIDs: (optional) list of ints: list of replicate IDs in experiment; needs to be specified only when replicates are to be excluded
        #ChannelNames: (optional) dictionary: dictionary of each preset channel on the microscope as a key and the desired name as value
    def __init__(self,exID=None,basedir=None,data=None,Deconvolution=False,\
                 maxPID=-1,reprange=10,replicateIDs=[],ChannelNames=None,rename=False):
        replicateIDs=replicateIDs
        self.Deconvolution = Deconvolution
        self.exID = exID
        self.basedir=basedir
        self.data=data
        self.maxPID=maxPID
        self.reprange=reprange
        self.ChannelNames=ChannelNames
        #if basedir is not specified it is calculated from exID and/or data
        if basedir==None:
            if data == None:
                for folder in os.listdir(data):
                    if str(exID) in folder:
                        self.basedir = os.path.join("Data", folder)
            else:
                for folder in os.listdir(data):
                    if str(exID) in folder:
                        self.basedir = os.path.join(data, folder)
        else: 
            self.basedir=basedir
        #loop through all replicate directories to find the maxPID and the replicateIDs
        for i in range(1,self.reprange):
            #update the path to the directory
            repdir = os.path.join(self.basedir,str(i))
            #check if directory exists; if not skip to next number
            if os.path.isdir(repdir)==False:
                continue
            #add the number to the replicateID list
            replicateIDs.append(i)
            #loop through all directories in replicate directory
            for fold in os.listdir(repdir):
                #check if it even is a directory; if not skip to next item
                if "." in fold:
                    continue
                #update to new directory
                conddir = os.path.join(repdir,fold)
                #loop through all files in directory...
                for file in os.listdir(conddir):
                    #...and get their
                    PID=None
                    try:
                        PID = int(file.split(".")[0])
                    except:
                        continue
                    
                    #if the PID is greater than the current max -> update
                    if PID > maxPID:
                        self.maxPID = PID
        self.replicateIDs = replicateIDs
        #if files need to be renamed they are afterwards
        if rename == True:
            for i in self.replicateIDs:
                self.rename_rep(i)
        return
    
    def split(self,imfile,imdir):
        #This function can be used to split a single image
        #Inputs:
            #imfile: str; specifies the name of the image to be split
            #imdir: str; specifies the directory where the image is found
        #To determine the name of the folder the ".nd2" needs to be removed
        fn = imfile.split(".")[0].replace(" ", "")
        #If deconvolution was performed the "Deconvolved" tag needs to be removed as well
        if "Deconvolved" in fn:
            fn=fn.split("Deconvolved")[0].split("-")[0]
        #The location to the new folder is created
        splitdir = os.path.join(imdir,fn)
        #If the folder already exists, it is interpreted that splitting was already performed, so the whole function stops
        #If splitting still needs to be performed, the folder should be deleted
        if os.path.isdir(splitdir)==True:
            print(fn+" already split")
            return
        #Else the folder is created
        else:
            os.mkdir(splitdir)
        #The ID of the next picture is calculated from th old max
        PID = self.maxPID+1
        #The picture is opened
        with nd2.ND2File(os.path.join(imdir,imfile)) as ndfile:
            #Metadata is read
            meta = ndfile.metadata
            #lenght of the x and y coordinates are read in um; these are assumed to be the same!
            xlen = meta.channels[0].volume.axesCalibration[0]
            #Names of the channels are read
            #If the names of the Channels are actually different than the ones in the microscope they are read from the dictionary
            if self.ChannelNames!=None:
                CN = {self.ChannelNames[meta.channels[ch].channel.name]: \
                      ch for ch in range(len(meta.channels))}
            else:
                CN = {meta.channels[ch].channel.name: ch for ch in range(len(meta.channels))}
            #If image is a series...
            if "P" in ndfile.sizes:
                #...loop through series
                for p in range(ndfile.sizes["P"]):
                    if "Z" in ndfile.sizes:
                        #if picture is a z Stack determine the size of the Z-step
                        zlen = meta.channels[0].volume.axesCalibration[2]
                        #save sliced image as tif
                        tiff.imsave(os.path.join(splitdir,str(PID)+".tif"),\
                                    ndfile.asarray()[p,:,:,:,:].astype('uint16'), imagej=True, \
                                        resolution=(1/xlen, 1/xlen),\
                                            metadata={'spacing':zlen, 'unit': 'um',"Channel":CN})
                    else:
                        tiff.imsave(os.path.join(splitdir,str(PID)+".tif"), \
                                    ndfile.asarray()[p,:,:,:].astype('uint16'), \
                                        imagej=True, resolution=(1/xlen, 1/xlen),\
                                            metadata={'unit': 'um',"Channel":CN})
                    #increment PID by one after each slice
                    PID = PID+1
                    
                #after the last slice the counter was incremented once too many times
                PID = PID-1
            else:
                if "Z" in ndfile.sizes:
                    zlen = meta.channels[0].volume.axesCalibration[2]
                    tiff.imsave(os.path.join(str(PID)+".tif"), ndfile.asarray().astype('uint16'),\
                                imagej=True, resolution=(1/xlen, 1/xlen),\
                                    metadata={'spacing': zlen, 'unit': 'um',"Channel":CN})
                else:
                    tiff.imsave(os.path.join(str(PID)+".tif"), ndfile.asarray().astype('uint16'), \
                                imagej=True, resolution=(1/xlen, 1/xlen),\
                                    metadata={'unit': 'um',"Channel":CN})
            #update the maximum ID
            self.maxPID=PID
            return
        
    def split_rep(self,rep):
        #This function can be used to split all images in a single replicate
        #Inputs:
            #rep: int; specifies the ID of the replicate
        #Update directory to replicate
        imdir = os.path.join(self.basedir,str(rep))
        #Loop through all files in replicate
        for imfile in tqdm(os.listdir(imdir),desc=str(rep)):
            #If file not a ".nd2" file -> skip
            if "nd2" not in imfile:
                continue
            #If Deconvolution is set to true -> skip those that are NOT Deconvolved
            if self.Deconvolution==True and "Deconvolved" not in imfile:
                continue
            #Split images individually as described above
            self.split(imfile,imdir)
        return
    
    def split_all(self):
        #This function can be used to split all images in a single experiment
        #Loop through all replicates
        for i in tqdm(self.replicateIDs,desc="Total"):
            print(i)
            #Split each replicate as described above
            self.split_rep(i)
        return
    
    def rename_rep(self,rep):
        #This function can be used to split all images in a single replicate
        #Inputs:
            #rep: int; specifies the ID of the replicate
        #Update directory to replicate
        repdir = os.path.join(self.basedir,str(rep))
        #Loop through all files in replicate
        for folder in tqdm(os.listdir(repdir),desc=str(rep)):
            #If file is not folder -> skip
            if "." in folder:
                continue
            else:
                #get directory of images
                imdir = os.path.join(repdir,folder)
                #loop through files and rename them after the corrsponding PID
                for file in os.listdir(imdir):
                    tPID = -1
                    #if the file already has a correct name it does not need to be renamed
                    try:
                        tPID = int(file.split(".")[0])
                    except:
                        os.rename(os.path.join(imdir,file), os.path.join(imdir,str(self.maxPID+1)+".tif"))
                        #update PID
                        self.maxPID=self.maxPID+1
        return
    
        



class segmentation(object):
    
    def __init__(self,df=pd.DataFrame(),exID=None,basedir=None,data=None,seg_nuc=False,\
                 model=None, xlen=0.1123,nuclei_channel="DAPI",cell_channel="GFP",\
                     t_mod=1,pixlensd = 0.3488,min_obj=5,intensity_channels=["aSyn_GFP"]\
                         ,maxCID=-1,reprange=10,replicateIDs=[],maxPID = -1,p_max=500,ch_dic=None):
        self.p_max=p_max
        self.df=df
        self.exID=exID
        self.data=data
        self.seg_nuc=seg_nuc
        self.model=model
        self.xlen=xlen
        self.nuclei_channel=nuclei_channel
        self.cell_channel=cell_channel
        self.t_mod=t_mod
        self.pixlensd=pixlensd
        self.min_obj=min_obj
        self.intensity_channels=intensity_channels
        self.maxCID=maxCID
        self.reprange=reprange
        self.replicateIDs=replicateIDs
        self.ch_dic={}
        #if basedir is not specified it is calculated from exID and/or data
        if basedir==None:
            if data == None:
                for folder in os.listdir(data):
                    if str(exID) in folder:
                        self.basedir = os.path.join("Data", folder)
            else:
                for folder in os.listdir(data):
                    if str(exID) in folder:
                        self.basedir = os.path.join(data, folder)
        else: 
            self.basedir=basedir
        #if nuclei are segmented first, the model is loaded first
        if seg_nuc==True:
            self.model = StarDist2D.from_pretrained('2D_versatile_fluo')
        #opens the Results table if it already exists
        if df.empty == True:
            resultspath = os.path.join(self.basedir,"Results.xlsx")
            if os.path.exists(resultspath) == True:
                self.df = pd.read_excel(resultspath)
                self.maxCID = self.df["CID"].max()
        if len(replicateIDs) == 0:
            for i in range(1,self.reprange):
                #update the path to the directory
                repdir = os.path.join(self.basedir,str(i))
                #check if directory exists; if not skip to next number
                if os.path.isdir(repdir)==False:
                    continue
                #add the number to the replicateID list
                replicateIDs.append(i)
            self.replicateIDs=replicateIDs
        if ch_dic!=None:
            for key in ch_dic:
                self.ch_dic[key]=ch_dic[key]
                self.ch_dic[ch_dic[key]]=key
        return
    
    def measure(self,labels,path):
        ncdfs=[pd.DataFrame(regionprops_table(labels,properties = ('label',"bbox")))]
        dcol = []
        for c in self.intensity_channels:
            intimg = getImage(path,c,cn=self.ch_dic[c])
            if len(intimg.shape) == 3:
                intimg = np.max(intimg,axis=0)
            ncdf = pd.DataFrame(regionprops_table(labels,intensity_image=intimg,properties = ('label',"bbox", "intensity_max","intensity_min","intensity_mean","area"), extra_properties=(intensity_median,)))
            ncdf.rename(columns=lambda x: x+"_"+c, inplace=True)
            dcol.append("label"+"_"+c)
            ncdf["CID"] = ncdf["label"+"_"+c]+self.maxCID
            ncdf.set_index("CID")
            ncdfs.append(ncdf)
        ncdf = pd.concat(ncdfs,axis = 1)
        #ncdf = ncdf.drop(labels=dcol, axis=1)
        return ncdf
    
    def segment_nuclei(self,cellpath):
        img = getImage(cellpath,self.nuclei_channel,cn=self.ch_dic[self.nuclei_channel])
        scaling_factor = self.xlen/self.pixlensd
        image_rescaled = rescale(img, scaling_factor, anti_aliasing=True, preserve_range = True)
        labelsN, _ = self.model.predict_instances(normalize(image_rescaled))
        nlL = resize(labelsN, img.shape, preserve_range=True, order=0, anti_aliasing=False).astype('uint8')
        return nlL
    
    def segment_cells(self,path):
        cells = getImage(path,self.cell_channel,cn=self.ch_dic[self.cell_channel])
        if len(cells.shape) == 3:
            cells = np.max(cells,axis=0)
        cellL = None
        if self.seg_nuc==True:
            nuclab = self.segment_nuclei(path)
            t = threshold_multiotsu(cells)
            binary = cells > round(t[0]*self.t_mod)
            binary = ndi.binary_fill_holes(binary)
            cellL = watershed(-cells, markers = nuclab, mask = binary)
        else:
            t = threshold_multiotsu(cells)
            binary1 = cells >= round(t[0]*self.t_mod)
            binary1 = ndi.binary_fill_holes(binary1)
            binary = np.zeros(binary1.shape, dtype = np.uint8)
            binary[binary1 == True] = 1
            distance = ndi.distance_transform_edt(binary)
            coords = peak_local_max(distance, footprint=np.ones((self.p_max,self.p_max)), labels=binary)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            cellL = watershed(-distance, markers = markers, mask = binary)
        
        cellL = morphology.remove_small_objects(cellL, 10**self.min_obj)
        if len(np.unique(cellL)) > 2:
            cellL = clear_border(cellL)
        return cellL
    
    def segment_all(self,replicate=None):
        for i in range(1,self.reprange):
            if replicate != None:
                if i != replicate:
                    continue
            #update the path to the directory
            repdir = os.path.join(self.basedir,str(i))
            #check if directory exists; if not skip to next number
            if os.path.isdir(repdir)==False:
                continue
            #loop through all directories in replicate directory
            for fold in os.listdir(repdir):
                #check if it even is a directory; if not skip to next item
                if "." in fold:
                    continue
                #update to new directory
                conddir = os.path.join(repdir,fold)
                #loop through all files in directory...
                for file in tqdm(os.listdir(conddir),desc=str(i)):
                    if ".tif" not in file:
                        continue
                    PID = int(file.split(".")[0])
                    if self.df.empty == False:
                        if PID in self.df["PID"].values:
                            continue
                    path = os.path.join(conddir,file)
                    cellL = self.segment_cells(path)
                    cdf = self.measure(cellL,path)
                    cdf["PID"] = PID
                    for key in self.ch_dic:
                        cdf["Channel_"+str(key)] = self.ch_dic[key]
                    roi = []
                    for l in cdf["label"].unique():
                        if l == 0:
                            continue
                        contour = find_contours(cellL==l, level=0.9999)[0]
                        ROI = ImagejRoi.frompoints(np.round(contour)[:, ::-1],name="Cell_"+str(l+self.maxCID))
                        roi.append(ROI)
                    roiwrite(os.path.join(conddir,str(PID)+".zip"),roi,mode="w")
                    cdf["ROI_path"] = os.path.join(conddir,str(PID)+".zip")
                    cdf["Replicate"] = str(i)
                    j = 1
                    for g in fold.split("_"):
                        cdf["Group"+str(j)] = g
                        j = j+1
                    cdf["path"] = path
                    if cdf.empty == False:
                        self.maxCID = cdf["CID"].max()
                    if self.df.empty == True:
                        self.df = cdf
                    else:
                        self.df = pd.concat([self.df,cdf],ignore_index=True)
        resultspath = os.path.join(self.basedir,"Results.xlsx")
        self.df.to_excel(resultspath)
        return
    
    def get_random_PID(self,reps,idn):
        PIDs = []
        for i in reps:
            #update the path to the directory
            repdir = os.path.join(self.basedir,str(i))
            #loop through all directories in replicate directory
            for fold in os.listdir(repdir):
                #check if it even is a directory; if not skip to next item
                if "." in fold:
                    continue
                #update to new directory
                conddir = os.path.join(repdir,fold)
                #loop through all files in directory...
                for file in os.listdir(conddir):
                    #...and get their PID
                    PID = int(file.split(".")[0])
                    #if the PID is greater than the current max -> update
                    PIDs.append(PID)
        ranID = np.random.choice(PIDs,idn,replace=False)
        return ranID
    
    def plot_segmentation(self,replicate=None,IDs=None,color="yellow",figsize=(16,24)):
        start_time = time.time()
        reps = self.replicateIDs
        if replicate != None:
            reps = [replicate]
        idlen = 0
        if IDs != None:
            idlen = len(IDs)
        if idlen < 3:
            newIDs = self.get_random_PID(reps,3-idlen)
            if IDs != None:
                IDs = [*IDs,*newIDs]
            else:
                IDs = newIDs
        print(IDs)
        plotdir={}
        for r in reps:
            #update the path to the directory
            repdir = os.path.join(self.basedir,str(r))
            #loop through all directories in replicate directory
            for fold in os.listdir(repdir):
                #check if it even is a directory; if not skip to next item
                if "." in fold:
                    continue
                #update to new directory
                conddir = os.path.join(repdir,fold)
                #loop through all files in directory...
                for file in os.listdir(conddir):
                    if ".tif" not in file:
                        continue
                    #...and get their PID
                    PID = int(file.split(".")[0])
                    if PID in IDs:
                        plotdir[PID]={}
                        cellL = self.segment_cells(os.path.join(conddir,file))
                        roi = []
                        for l in np.unique(cellL):
                            if l == 0:
                                continue
                            contour = find_contours(cellL==l, level=0.9999)[0]
                            roi.append(ImagejRoi.frompoints(np.round(contour)[:, ::-1]))
                        plotdir[PID]["ROI"]=roi
                        plotdir[PID]["path"]=os.path.join(conddir,file)
                        print(str(PID)+": "+str(time.time()-start_time))
                        start_time = time.time()
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        subfigs = fig.subfigures(1,3)
        yC = 0
        for PID in plotdir:
            ax0 = subfigs[yC].subplots(1, 1, sharey=True)
            img = getImage(plotdir[PID]["path"],self.cell_channel,cn=self.ch_dic[self.cell_channel])
            img = equalize_adapthist(img)
            if len(img.shape) == 3:
                img = np.max(img,axis=0)
            ax0.imshow(img, cmap='gray')
            rois = plotdir[PID]["ROI"]
            if not isinstance(rois, list):
                rois = [rois]
            for ROI in rois:
                ROI.plot(ax0, lw=1,color=color)
            ax0.tick_params(bottom=False, top=False, left=False, right=False)
            ax0.set_title(str(PID), fontsize=11)
            yC = yC+1
        plt.show()
        print("Plotting"+str(time.time()-start_time))
        return

class detection(object):
    
    def __init__(self,df=pd.DataFrame(),odf=pd.DataFrame(),exID=None,basedir=None,data=None,sigma1=0.2,radius=15,sigma2=2,channels="ALL",ntol=30,t_mod=1,max_OID=-1,ch_dic=None):
        self.df=df
        self.exID=exID
        self.data=data
        self.sigma1=sigma1
        self.radius=radius
        self.sigma2=sigma2
        self.channels=channels
        self.ntol=ntol
        self.t_mod=t_mod
        self.max_OID=max_OID
        self.odf = odf
        self.ch_dic = {}
        #if basedir is not specified it is calculated from exID and/or data
        if basedir==None:
            if data == None:
                for folder in os.listdir(data):
                    if str(exID) in folder:
                        self.basedir = os.path.join("Data", folder)
            else:
                for folder in os.listdir(data):
                    if str(exID) in folder:
                        self.basedir = os.path.join(data, folder)
        if df.empty == True:
            resultspath = os.path.join(self.basedir,"Results.xlsx")
            if os.path.exists(resultspath) == True:
                self.df = pd.read_excel(resultspath)
        if ch_dic!=None:
            for key in ch_dic:
                self.ch_dic[key]=ch_dic[key]
                self.ch_dic[ch_dic[key]]=key
                
    def getCellROI(self,CID):
        roi_path = self.df["ROI_path"][self.df["CID"]==CID].values[0]
        ROIs = ImagejRoi.fromfile(roi_path)
        for r in ROIs:
            if r.name == "Cell_"+str(CID):
                return r 
    
    def denoise(self,CID,ch):
        #The first function implements the denoising itself
        #First it opens the whole image, then reads the bounding box of the specified cell and then crops it to the ROI to speed up computation
        #Denoising is then performed on the image or each z-Slice if a 3d image is provided.
        #The following steps are performed:
            #1. Gaussian filtering with a small sigma to remove "salt and pepper noise"
            #2. Rolling ball background estimation with a small ball size
            #3. Background gets smoothed by convolving a gaussian filter with a large sigma to retain details
            #4. Background subtraction
        #Inputs:
            #sigma1: sigma of first guassian filter
            #radius: radius of rolling ball
            #sigma2: sigma of second gaussian filter
        #Output:
            #Cropped and denoised image of a single cell
        #Opens the correct image
        path = self.df["path"][self.df["CID"]==CID].values[0]
        ROI = self.getCellROI(CID)
        img = getImage(path,ch,cn=self.ch_dic[ch])
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
                gimg = gaussian(cimg[st,:,:],self.sigma1)
                #Perform rolling ball background estimation; Low radius will remove large amounts of cytoplasmic detail but keep the peaks intact
                bg = rolling_ball(gimg, radius=self.radius)
                #Gaussian smoothing of the background
                bg = gaussian(bg,self.sigma2)
                #Subtract background
                bgsub = gimg-bg
                #Store the denoised slice in the prevously created array
                newimg.append(bgsub)
            #return the slices after putting them back together
            return np.array(newimg)
        #else image is 2d
        else:
            #Apply first gaussian smoothing
            #Low sigma leads to little change in detail but removal of fine high frequency noise
            gimg = gaussian(cimg,self.sigma1)
            #Perform rolling ball background estimation; Low radius will remove large amounts of cytoplasmic detail but keep the peaks intact
            bg = rolling_ball(gimg, radius=self.radius)
            #Gaussian smoothing of the background
            bg = gaussian(bg,self.sigma2)
            #Subtract background
            bgsub = gimg-bg
            return bgsub
        
    def detect_objects(self,CID,ch):
        c = self.ch_dic[ch]
        denoised = self.denoise(CID,ch)
        odfs = []
        ROI = self.getCellROI(CID)
        ROIcoords = np.subtract(ROI.coordinates(),np.array([[ROI.left,ROI.top]]))
        if len(denoised.shape) == 3:
            for st in range(denoised.shape[0]):
                #label = grid_points_in_poly(denoised[st,:,:].shape,ROIcoords)
                #label = label>0
                pimp = rescale_intensity(denoised[st,:,:],out_range="uint8")
                #pimp[label==False] = 0
                local_max = find_local_maxima(pimp)
                y, x, regs = find_maxima(pimp,local_max,self.ntol)
                coo = np.array([x,y]).T
                coomask = points_in_poly(coo,ROIcoords)
                odf = pd.DataFrame(data={"x":x,"y":y,"InCytoplsm":coomask})
                odf["CID"] = CID
                odf["channel"] = ch
                odf["channelID"] = c
                odf["z"] = st
                odf["x"] = odf["x"]+ROI.left
                odf["y"] = odf["y"]+ROI.bottom
                odfs.append(odf)
            return pd.concat(odfs,ignore_index=True),x,y,denoised
        else:
            #label = grid_points_in_poly(denoised[st,:,:].shape,ROIcoords)
            #label = label>0
            #pimp[label==False] = 0
            pimp = rescale_intensity(denoised,out_range="uint8")
            local_max = find_local_maxima(pimp)
            y, x, regs = find_maxima(pimp,local_max,self.ntol)
            coo = np.array([x,y]).T
            coomask = points_in_poly(coo,ROIcoords)
            odf = pd.DataFrame(data={"x":x,"y":y,"InCytoplsm":coomask})
            odf["CID"] = CID
            odf["channel"] = ch
            odf["channelID"] = c
            odf["int"] = ndi.map_coordinates(pimp,np.array([pimp.shape[0]-y,x]), order=1)
            odf["x"] = odf["x"]+ROI.left
            odf["y"] = odf["y"]+ROI.bottom
            odfs.append(odf)
        return pd.concat(odfs,ignore_index=True),x,y,denoised
    
    def detect_objects_indiv(self,PID,ch):
        c = self.ch_dic[ch]
        
        denoised = self.denoise_indiv(PID,ch)
        
        odfs = []
        #label = grid_points_in_poly(denoised[st,:,:].shape,ROIcoords)
        #label = label>0
        #pimp[label==False] = 0
        start_time=time.time()
        pimp = rescale_intensity(denoised,out_range="uint8")
        local_max = find_local_maxima(pimp)
        y, x, regs = find_maxima(pimp,local_max,self.ntol)
        print(start_time-time.time())
        start_time=time.time()
        odf = pd.DataFrame(data={"x":x,"y":y})
        return odf,x,y,denoised
        
    def detect_all(self,replicate=None,channels=[]):
        df = self.df
        if replicate != None:
            df = self.df[self.df["Replicate"]==replicate]
        for PID in tqdm(df["PID"].unique()):
            odfs = []
            tsvpath = os.path.join(os.path.split(df["path"][df["PID"]==PID].values[0])[0],str(df["PID"][df["PID"]==PID].values[0])+".tsv")
            for CID in df["CID"][df["PID"]==PID].unique():
                if channels == []:
                    for col in list(df.columns):
                        if "Channel" in col:
                            channels.append(df[col][df["CID"]==CID].values[0])
                for ch in channels:
                    new_odf,_,_,_ = self.detect_objects(CID, ch)
                    odfs.append(new_odf)
            pd.concat(odfs).to_csv(tsvpath, sep="\t")
            self.df.loc[self.df.PID == PID, "coord_path"] = tsvpath
        self.df.to_excel(os.path.join(self.basedir,"Results.xlsx"))
        
    def denoise_indiv(self,PID,ch):
        #The first function implements the denoising itself
        #First it opens the whole image, then reads the bounding box of the specified cell and then crops it to the ROI to speed up computation
        #Denoising is then performed on the image or each z-Slice if a 3d image is provided.
        #The following steps are performed:
            #1. Gaussian filtering with a small sigma to remove "salt and pepper noise"
            #2. Rolling ball background estimation with a small ball size
            #3. Background gets smoothed by convolving a gaussian filter with a large sigma to retain details
            #4. Background subtraction
        #Inputs:
            #sigma1: sigma of first guassian filter
            #radius: radius of rolling ball
            #sigma2: sigma of second gaussian filter
        #Output:
            #Cropped and denoised image of a single cell
        #Opens the correct image
        path = self.df["path"][self.df["PID"]==PID].values[0]
        img = getImage(path,ch,cn=self.ch_dic[ch])
        # Images are cropped
        cimg = img
        #CHeck if image is 3d
        if len(img.shape)==3:
            #Since each slice of a 3d image gets processed individually the slices have to be temporarily stored in an array and then put together afterwards
            newimg = []
            #Loop through all slices
            for st in range(img.shape[0]):
                #Apply first gaussian smoothing
                #Low sigma leads to little change in detail but removal of fine high frequency noise
                gimg = gaussian(cimg[st,:,:],self.sigma1)
                #Perform rolling ball background estimation; Low radius will remove large amounts of cytoplasmic detail but keep the peaks intact
                bg = rolling_ball(gimg, radius=self.radius)
                #Gaussian smoothing of the background
                bg = gaussian(bg,self.sigma2)
                #Subtract background
                bgsub = gimg-bg
                #Store the denoised slice in the prevously created array
                newimg.append(bgsub)
            #return the slices after putting them back together
            return np.array(newimg)
        #else image is 2d
        else:
            #Apply first gaussian smoothing
            #Low sigma leads to little change in detail but removal of fine high frequency noise
            gimg = gaussian(cimg,self.sigma1)
            #Perform rolling ball background estimation; Low radius will remove large amounts of cytoplasmic detail but keep the peaks intact
            bg = rolling_ball(gimg, radius=self.radius)
            #Gaussian smoothing of the background
            bg = gaussian(bg,self.sigma2)
            #Subtract background
            bgsub = gimg-bg
            return bgsub
        
    def detect_indiv(self,replicate=None,channels=[]):
        df = self.df
        if replicate != None:
            df = self.df[self.df["Replicate"]==replicate]
        for PID in tqdm(df["PID"].unique()):
            odfs = []
            tsvpath = os.path.join(os.path.split(df["path"][df["PID"]==PID].values[0])[0],str(df["PID"][df["PID"]==PID].values[0])+".tsv")
            if channels == []:
                for col in list(df.columns):
                    if "Channel" in col:
                        channels.append(df[col][df["PID"]==PID].values[0])
            for ch in channels:
                new_odf,_,_,_ = self.detect_objects_indiv(PID, ch)
                odfs.append(new_odf)
            pd.concat(odfs).to_csv(tsvpath, sep="\t")
            self.df.loc[self.df.PID == PID, "coord_path"] = tsvpath
        self.df.to_excel(os.path.join(self.basedir,"Results.xlsx"))
                
        
    
    def plot_detection(self,replicate=None,IDs=None,color="yellow",figsize=(16,24),channel="aSyn-GFP"):
        start_time = time.time()
        df = self.df
        if replicate != None:
            df = df[df["Replicate"]==replicate]
        idlen = 0
        if IDs != None:
            idlen = len(IDs)
        if idlen < 2:
            if replicate==None:
                newIDs = np.random.choice(df["CID"].values,2-idlen)
                if IDs != None:
                    IDs = [*IDs,*newIDs]
                else:
                    IDs = newIDs
        print(IDs)
        plotdir={}
        for CID in IDs:
            plotdir[CID]={}
            _,objROIs,denoised = self.detect_objects(CID,channel)
            plotdir[CID]["image"]=denoised
            plotdir[CID]["ROI"]=objROIs
            plotdir[CID]["path"]=self.df["path"][self.df["CID"]==CID].values[0]
            print(str(CID)+": "+str(time.time()-start_time))
            start_time = time.time()
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        subfigs = fig.subfigures(1,2)
        yC = 0
        for PID in plotdir:
            ax0 = subfigs[yC].subplots(1, 1, sharey=True)
            img = plotdir[PID]["image"]
            img = equalize_adapthist(img)
            if len(img.shape) == 3:
                img = np.max(img,axis=0)
            ax0.imshow(self.crop(img,plotdir[PID]["CellROI"]), cmap='gray')
            rois = plotdir[PID]["ROI"]
            if not isinstance(rois, list):
                rois = [rois]
            for ROI in rois:
                roi = ImagejRoi.frombytes(ROI)
                roi.plot(ax0, lw=1,color=color)
            ax0.tick_params(bottom=False, top=False, left=False, right=False)
            ax0.set_title(str(PID), fontsize=11)
            yC = yC+1
        plt.show()
        print("Plotting"+str(time.time()-start_time))
        return
    
    def detect_simple(self,replicate=None,channel="aSyn_GFP"):
        df = self.df
        data = {"CID":[],"ObjC":[]}
        if replicate != None:
            df = self.df[self.df["Replicate"]==replicate]
        cID = self.ch_dic[channel]
        for PID in tqdm(df["PID"].unique()):
            path = df["path"][df["PID"]==PID].values[0]
            roi_path = df["ROI_path"][df["PID"]==PID].values[0]
            img = tiff.imread(path)[cID,:,:]
            ROIs = ImagejRoi.fromfile(roi_path)
            t = np.median(img)*3
            coordinates = peak_local_max(img, min_distance=20,threshold_abs = t)
            for ROI in ROIs:
                print(ROI.coordinates)
                coomask = points_in_poly(coordinates,ROI.coordinates)
                data["CID"].append(int(ROI.name.split("_")[1]))
                data["CID"].append(np.sum(coomask))
        self.df = df.merge(pd.DataFrame(data=data),on = "CID")
        self.df.to_excel(os.path.join(self.basedir,"Results.xlsx"),ignore_index=True)
    
    def return_info(self):
        return self.df, self.ch_dic, self.basedir
            

dataname = "D:\Raphael\Data"
basedir = "D:/Raphael/Data/3. Plasmid moves cargo into Lysosomes"
exID = 3
CD={0:"aSyn_GFP",1:"aSyn_AB",2:"LAMP1/LC3"}
#registration(basedir=basedir,Deconvolution=True).split_rep(2)
















