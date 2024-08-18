# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:30:58 2023

@author: Raphael
"""

import os
import nd2
import numpy as np

def readimages(rawdir, splitdir, ID, NDID,PID,rep):
    cc = 0
    import os
    import tifffile as tiff
    import nd2
    data = {"ID":[], "PID":[], "Channel": [], "ChID": [], "xlen": [], "ylen":[], "zlen": [], "Group1":[], "Group2":[], "NDID":[],"Series":[],"Stack":[],"Rep":[], "Path": []}
    for ndimg in os.listdir(rawdir):
        if "Deconvolved" in os.listdir(rawdir) and "Deconvolved" not in ndimg:
            continue
        iml = os.path.join(rawdir, ndimg)
        with nd2.ND2File(iml) as ndfile:
            meta = ndfile.metadata
            series = ["None"]
            stack = ["None"]
            if len(ndfile.shape) > 3:
                series = range(ndfile.shape[0])
                stack = range(ndfile.shape[1])
            for s in series:
                for st in stack:
                    for i in range(len(meta.channels)):
                        chan = meta.channels[i].channel.name
                        ch = ""
                        if chan in ["594"]:
                            ch = "ConcanavalinA"
                        elif "DAPI" in chan:
                            ch = "DAPI"
                        elif chan in ["GFP"]:
                            ch = "aSyn_GFP"
                        elif chan in ["647"]:
                            ch = "aSyn_AB"
                        elif "555" in chan:
                            ch = "LAMP1"
                        elif "568" in chan:
                            ch = "LC3"
                        else:
                            continue
                        #img = ndfile.asarray(PID)
                        #img = img[:,:,i,:,:]
                        #tiff.imwrite(os.path.join(splitdir,str(ID))+".tif", img)
                        xlen, ylen, zlen = meta.channels[i].volume.axesCalibration
                        data["ID"].append(str(ID))
                        data["PID"].append(str(PID))
                        data["NDID"].append(NDID)
                        data["Channel"].append(str(ch))
                        data["ChID"].append(i)
                        data["xlen"].append(xlen)
                        data["ylen"].append(ylen)
                        data["zlen"].append(zlen)
                        data["Series"].append(s)
                        data["Stack"].append(st)
                        data["Group1"].append(ndimg.split("_")[0])
                        data["Path"].append(iml)
                        if "Deconvolved" in ndimg:
                            data["Group2"].append(ndimg.split("_")[1].split(" ")[0])
                        else:
                            data["Group2"].append(ndimg.split("_")[1].split(".")[0])
                        data["Rep"].append(rep)
                        ID = ID+1
                    NDID+=1
                PID = PID+1
    return data

def splitimages(experiment):
    import os
    import pandas as pd
    basedir = getimfolder(experiment)
    for i in range(5):
        if str(i) in os.listdir(basedir):
            rep = i
            rawdir = os.path.join(basedir,str(rep))
            splitdir = os.path.join(basedir,"Images")
            info = os.path.join(basedir, "Info.xlsx")
            if os.path.exists(info) == True:
                old_df = pd.read_excel(info)
                maxrep = max(old_df["Rep"])
                if maxrep >= i:
                    continue
                ID = max(old_df["ID"])
                NDID = max(old_df["NDID"])
                PID = max(old_df["PID"])
                data = readimages(rawdir, splitdir, ID, NDID,PID,rep)
                new_df = pd.DataFrame(data = data)
                print(len(new_df))
                df = pd.concat([old_df,new_df])
                df = df.drop(df[df["Channel"] == "DIA Confocal"].index)
            else:
                ID = 0
                data = readimages(rawdir, splitdir, ID, 0,0,rep)
                df = pd.DataFrame(data = data)
                print(len(df))
                df = df.drop(df[df["Channel"] == "DIA Confocal"].index)
            df.to_excel(info,index=False)
    

def getimfolder(experiment):
    import os
    basedir = ""
    data = "E:\Raphael\Data"
    for folder in os.listdir(data):
        if str(experiment) in folder:
            basedir = os.path.join(data, folder)
    return basedir

def getImage(df, PID,ch, basedir):
    iml = str(df["Path"][df["PID"]==PID].values[0])
    with nd2.ND2File(iml) as ndfile:
        img = ndfile.to_dask()
        i = df["ChID"][df["Channel"]==ch][df["PID"]==PID].values[0]
        if len(img.shape) == 3:
            img = img[i,:,:]
            return img
        else:
            p = df["Series"][df["PID"]==PID].values[0]
            img = img[p,:,i,:,:]
            return img
        #img = np.array(img)
                