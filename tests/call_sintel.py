import numpy as np
import sys
sys.path.append('/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN')
import tflib as lib
import tflib.save_images
import tflib.SINTELdataDesktop as sintel

BATCH_SIZE = 10 # Batch size
TARGET_SIZE = 100
OUT_DIM = TARGET_SIZE*TARGET_SIZE*3

gen = sintel.load_train_gen(BATCH_SIZE,(TARGET_SIZE, TARGET_SIZE, 3))
_data, _ = next(gen)
print(_data.shape)
            
images = _data[0]
#print(type(images))   np ndarray
#print(images.dtype)  uint8

image1= images[0:OUT_DIM]
image2= images[OUT_DIM-1:]
print(image1.shape)  # (outdim,)
image1 = image1.reshape(TARGET_SIZE,TARGET_SIZE,3)

outpath = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/data/gentest/sample_sintel_"
tflib.save_images.save_images(image1.reshape((1,3,TARGET_SIZE,TARGET_SIZE)), outpath+"5.jpg")

