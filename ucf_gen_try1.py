
import os, sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
import tflib as lib
import tflib.save_images
import tflib.plot
import tflib.ucf101
import tflib.processor as proc
import tflib.UCFdata
from keras.preprocessing.image import img_to_array


BATCH_SIZE = 10 # Batch size
gen = lib.UCFdata.load_train_gen(BATCH_SIZE)

_data, _ = next(gen)
            
image1 = _data[0]
print(image1.shape)
img_arr = img_to_array(image)
x = (img_arr / 255.).astype(np.float32)
# image1 = proc.process_image(_data[0])
# image1  = np.transpose(image1, [2,0,1])
outpath = "/home/linkermann/opticalFlow/opticalFlowGAN/data/gentest/sample"
tflib.save_images.save_images(image1.reshape((1,3,32,32)), outpath+str(iteration)+".jpg")
