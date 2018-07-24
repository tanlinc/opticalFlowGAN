import numpy as np
import tflib as lib
import tflib.save_images
import tflib.processor as proc
import tflib.UCFdataDesktop
from keras.preprocessing.image import img_to_array

BATCH_SIZE = 10 # Batch size
gen = lib.UCFdataDesktop.load_train_gen(BATCH_SIZE)

_data, _ = next(gen)
            
image1 = _data[0]
#print(image1.shape) # (3072,)
image1 = image1.reshape(32,32,3)
image1  = np.transpose(image1, [2,0,1])
outpath = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/data/gentest/sample_bla_"
tflib.save_images.save_images(image1.reshape((1,3,32,32)), outpath+".jpg")
