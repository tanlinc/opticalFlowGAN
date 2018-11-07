import os, sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
import tflib as lib
import tflib.save_images
import tflib.flow_handler as fh
import tflib.SINTELdata as sintel

BATCH_SIZE = 64 # Batch size
IM_DIM = 32 # number of pixels along x and y (square assumed)
OUTPUT_DIM = IM_DIM*IM_DIM*3 # Number of pixels (3*32*32) - rgb color
OUTPUT_DIM_FLOW = IM_DIM*IM_DIM*2 # Number of pixels (2*32*32) - uv direction
lib.print_model_settings(locals().copy())

# Dataset iterators
gen = sintel.load_train_gen(BATCH_SIZE, (IM_DIM,IM_DIM,3), (IM_DIM,IM_DIM,2)) # batch size, im size, im size flow

# For generating samples: define conditional input
_data, _flow  = next(gen) # fixed_cond_samples, fixed_flow_samples = next(gen)  # shape: (batchsize, 3072) 
fixed_cond_data_int = _data[:,0:2*OUTPUT_DIM] # earlier frames as condition, cond samples shape (64,3*3072)
fixed_viz_data_int = _data[:,OUTPUT_DIM:2*OUTPUT_DIM]

# viz cond data
images = fixed_viz_data_int.reshape(BATCH_SIZE, IM_DIM,IM_DIM,3)
outpath = "/home/linkermann/opticalFlow/opticalFlowGAN/data/gentest/"
tflib.save_images.save_images(images.reshape((BATCH_SIZE,3,IM_DIM,IM_DIM)), outpath+"condvizbatch.jpg")

fixed_real_data =_flow[:,OUTPUT_DIM_FLOW:]	 # later flow for discr, flow samples shape (64,2048)

init_op = tf.global_variables_initializer()  	# op to initialize the variables.

# Train loop
with tf.Session() as session:
    session.run(init_op)

    # samples_255 = np.zeros((2*BATCH_SIZE, IM_DIM, IM_DIM, 3))
    real_flowimages = []
    for i in range(0, BATCH_SIZE):
        real_flowimg = [] # reset to be sure
        real_flowimg = fh.computeFlowImg(fixed_real_data[i,:].reshape((IM_DIM,IM_DIM,2)))  # (32, 32, 3) # now color img!! :)
        # real_flowimage_T = np.transpose(real_flowimg, [2,0,1])  #  (3, 32, 32)
        # real_flowimages.append(real_flowimage) # np.asarray...      
        real_flow = real_flowimg.astype('int32') # diff numbers 0..255
        # samples_255[2*i+1,:,:,:] = real_flow 
        images = np.insert(images, i*2+1, real_flow, axis=0)

        # last_frame = images[i,:,:,:].astype('int32')  
        #last_frame = last_frame.reshape(IM_DIM,IM_DIM,3)
        #last_frame_transposed = last_frame.reshape(3,IM_DIM,IM_DIM) # (3072,)
        # samples_255[2*i,:,:,:] = last_frame # last frame left of generated sample
# samples_255= np.insert(samples_255, i*2, fixed_cond_data_int[i],axis=0)

    lib.save_images.save_images(samples_255.reshape((2*BATCH_SIZE, 3, IM_DIM, IM_DIM)), 'cond_frames_batch.jpg')    		

