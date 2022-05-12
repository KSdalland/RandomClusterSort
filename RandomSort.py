
from cmath import inf, nan

import cv2
import numpy as np
import os
import sys

import face_recognition

# faceArray=face_recognition.face_encodings(im, boxes)[0]
# faceData.append({"iname": img, "BBOX": [int(x), int(y), int(w), int(h)],"Features": faceArray})

def RandomSort(K, NumberOfFaces, FaceData):
    NumFaces=NumberOfFaces
    ObigDiff=1e10
    trials=5000
    count=0
    bestClustering=[{}]*K
    randy=np.random.randint(low = 0, high = NumFaces, size = trials*K)
    IMAGnum=2
    tempResults=[{"cluster_no": 0, "elements": [], "features": [], "BBOX": []}]*K
    # RANDOM SORT
    for i in range(trials):
        print("   ...",count+1," \ ", trials,end="\r")
        
        clusterCenters=[]
        for k in range(K):
            clusterCenters.append((FaceData[int(randy[count*k])]["Features"]))
            tempResults[k]={"cluster_no": k, "elements": [], "features": [], "BBOX": []}

        bigDiff=0
        for face in FaceData:
            oldDskew=0
            Olddiff=1e10
            faceArray=face["Features"]
            Cimage=face["iname"]
            boundbox=face["BBOX"]
            
            for k in range(K):
                Mu=clusterCenters[k]
                diff=np.sum(abs(Mu-faceArray))

                if diff<Olddiff:
                    Olddiff=diff
                    clusterNum=k
                    oldDskew=Olddiff

            tempResults[clusterNum]["elements"].append(Cimage)
            tempResults[clusterNum]["features"].append(faceArray)
            tempResults[clusterNum]["BBOX"].append(boundbox)

            bigDiff+=oldDskew

        for k in range(K):        
            check3=len(tempResults[k]["elements"])
            #print(check3)
            
            if check3<IMAGnum:
                bigDiff+=1e10
                

        if ObigDiff>bigDiff:
            ObigDiff=bigDiff

            for k in range(K): 
                bestClustering[k]=tempResults[k]
                check3=len(tempResults[k]["elements"])
                #print(check3)

        count+=1

        if i==trials-1:
            break
    return bestClustering
