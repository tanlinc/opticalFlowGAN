import numpy as np
import sys
sys.path.append('/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN')
import tflib as lib
import tflib.save_images
import tflib.SINTELdataDesktop as sintel
import matplotlib.pyplot as plt

BATCH_SIZE = 10 # Batch size
TARGET_SIZE = 64
OUT_DIM = TARGET_SIZE*TARGET_SIZE*3
OUT_DIM_FLOW = TARGET_SIZE*TARGET_SIZE*2

gen = sintel.load_train_gen(BATCH_SIZE,(TARGET_SIZE, TARGET_SIZE, 3))
_data, _flow = next(gen)
# images: (n, 6144) -- 3072 + 3072 = two images for 32
# flows: (n, 4096) -- 2048 + 2048 = two flows for 32
            
images = _data[0]  # np ndarray, uint8 

image1= images[0:OUT_DIM]  # (outdim,) # (3072,) for 32
image2= images[OUT_DIM-1:]
image1 = image1.reshape(TARGET_SIZE,TARGET_SIZE,3)
outpath = "..."
tflib.save_images.save_images(image1.reshape((1,3,TARGET_SIZE,TARGET_SIZE)), outpath+"8.jpg")

flows = _flow[0]
flow1 = flows[0:OUT_DIM_FLOW]  # (2048,) for 32
flow2 = flows[OUT_DIM_FLOW:]   #(2048,) for 32
flow1 = flow1.reshape(TARGET_SIZE,TARGET_SIZE,2)
flow2 = flow2.reshape(TARGET_SIZE,TARGET_SIZE,2)

