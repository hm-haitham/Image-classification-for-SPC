import os 
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MANTIS import MANITS_Camera 
from MDSplus import Connection



## >> ssh -NT4 cours24@lac.epfl.ch -L 5555:mantis4:8000 
## in a terminal
con = Connection('localhost:5555')
meta=pd.read_csv("meta.csv")

# Setting up the resizing parameters
IMG_HEIGHT=128
IMG_WIDTH=128
# We create a single black frame that we can concatenate on later
all_shots=np.zeros((1,128,128,1))
all_labels=[0]
frames_per_shot={}
for index, row in meta.iterrows():
    if (index!=23):  # This video was wrongfully IDed
        print(row["shot number"])
        counter=0
        for j in range(1,11):       
            print("Working on channel ",j)

            # Upload the footage

            # There is no footage for these sepecific cameras
            if((index!=18) | (j!=10)):
                if (str(j) in row["do not use cameras"]):
                    print ("camera {cam} is unusable".format(cam=j))

                else:

                    cam=MANITS_Camera(row["shot number"],j)

                    # pull the images, this will be our input
                    start_time=max(0,cam.times.vals.min())
                    finish_time=cam.times.vals.max()
                    frames=cam.times.vals

                    if row["other"]=="don't use frames after end of leg":
                        finish_time=row["leg end"]

                    # We will take every 10th frame as a subsample, this is useful to reduce overfitting and to accelerate
                    # the loading process
                    shots=frames[(frames>start_time) & (frames<=finish_time)][0::10]

                    # the function "images" may create a runtime error on mds, so we need to retry if it does
                    exception_counter=0
                    while True:
                        try:
                            imgs, tims =cam.images(shots)
                        except Exception as e:
                            if(exception_counter<10):
                                exception_counter+=1
                                print("Server timed out , retrying ...")
                                time.sleep(10)
                                continue
                            else:
                                print(e)
                        break


                    #resize the imgs
                    imgs=cv2.resize(imgs, dsize=(IMG_HEIGHT,IMG_WIDTH), interpolation=cv2.INTER_NEAREST)

                    #This is a preprocessing transformation to create the tensor later 
                    imgs=np.expand_dims(np.moveaxis(imgs,-1,0),axis=3)

                    # label the images
                    labels=tims.copy()
                    labels[(row["leg start time (s)"]<=labels) & (labels<=row["leg end"])]=1
                    labels[labels!=1]=0
                    labels=labels.T[0]

                    #concat
                    all_shots=np.concatenate((all_shots,imgs),axis=0)
                    all_labels=np.append(all_labels,labels)
                    print("collected shots : ",all_shots.shape[0])
        print("Collected shots up to this video  :", all_shots.shape[0])
        frames_per_shot[row["shot number"]]=all_shots.shape[0]
keys = np.fromiter(frames_per_shot.keys(), dtype=int)
vals = np.fromiter(frames_per_shot.values(), dtype=int)
np.savez_compressed('shots', shots=all_shots, labels=all_labels,keys=keys,vals=vals)
print("Operation complete, images saved as npz file in \"shots\"")