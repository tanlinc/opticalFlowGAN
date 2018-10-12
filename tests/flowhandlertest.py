import numpy as np
import sys
sys.path.append('/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN')
import tflib as lib
from tflib.save_images import save_images
import tflib.SINTELdata as sintel
import tflib.flow_handler as fh

BATCH_SIZE = 10 # Batch size
TARGET_SIZE = 300
OUT_DIM_FLOW = TARGET_SIZE*TARGET_SIZE*2

gen = sintel.load_train_gen(BATCH_SIZE,(TARGET_SIZE, TARGET_SIZE, 3), (TARGET_SIZE, TARGET_SIZE, 2))
_data, _flow = next(gen)
# images: (n, 6144) -- 3072 + 3072 + 3072 = three images for 32
# flows: (n, 4096) -- 2048 + 2048 = two flows for 32
            
outpath = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/data/gentest/"

flows = _flow[0] # first from batch
flow1 = flows[0:OUT_DIM_FLOW] # (2048,) for 32
flow2 = flows[OUT_DIM_FLOW:] # (2048,) for 32
flow1 = flow1.reshape(TARGET_SIZE,TARGET_SIZE,2)
flow2 = flow2.reshape(TARGET_SIZE,TARGET_SIZE,2)

# save flow               # write_flo_file(flow, filename)  
# filename must be string and end in .flo. Flow must be in (w,h,2) format
# fh.write_flo_file(flow1, outpath+'sample_sintel_flow.flo')  
# load flow from file     # read_flo_file(filename)
# flowfile = fh.read_flo_file(outpath+"sample_sintel_flow.flo")

# show flow               # computeImg(flow)
flowimg = fh.computeFlowImg(flow1)    # (200, 200, 3) # now color img!! :)
flowimage_T = np.transpose(flowimg, [2,0,1])  #  (3, 200, 200)
save_images(flowimage_T.reshape((1,3,TARGET_SIZE,TARGET_SIZE)), outpath+"flowsintel10.jpg")
