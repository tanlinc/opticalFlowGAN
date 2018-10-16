import numpy as np
import sys
sys.path.append('/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN')
import tflib as lib
from tflib.save_images import save_images
#import tflib.SINTELdata as sintel
import tflib.flow_handler as fh

BATCH_SIZE = 10 # Batch size
TARGET_SIZE = 300
OUT_DIM_FLOW = TARGET_SIZE*TARGET_SIZE*2
            
outpath = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/data/gentest/"

flow = np.zeros((TARGET_SIZE,TARGET_SIZE,2))
flow[:,:,0] = ...
flow[:,:,1] = ...

# show flow               # computeImg(flow)
flowimg = fh.computeFlowImg(flow)    # (TARGET_SIZE, TARGET_SIZE, 3) # now color img!! :)
flowimage_T = np.transpose(flowimg, [2,0,1])  #  (3, TARGET_SIZE, TARGET_SIZE)
save_images(flowimage_T.reshape((1,3,TARGET_SIZE,TARGET_SIZE)), outpath+"flowfieldviz.jpg")
