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
save_path = "/home/linkermann/opticalFlow/opticalFlowGAN/data/tests"
lib.print_model_settings(locals().copy())

cond_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 2*OUTPUT_DIM]) # cond input for G and D, 2 frames!
cond_data = 2*((tf.cast(cond_data_int, tf.float32)/255.)-.5) #normalized [-1,1]!
real_data =  tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM_FLOW]) #already float, normalized [-1,1]!

# Dataset iterators
gen = sintel.load_train_gen(BATCH_SIZE, (IM_DIM,IM_DIM,3), (IM_DIM,IM_DIM,2)) # batch size, im size, im size flow

# For generating samples: define fixed noise and conditional input
fixed_cond_samples, fixed_flow_samples = next(gen)  # shape: (batchsize, 3072) 
fixed_cond_data_int = fixed_cond_samples[:,0:2*OUTPUT_DIM] # earlier frames as condition, cond samples shape (64,3*3072)
fixed_real_data = fixed_flow_samples[:,OUTPUT_DIM_FLOW:]	 # later flow for discr, flow samples shape (64,2048)
fixed_real_data_norm01 = tf.cast(fixed_real_data+1.0, tf.float32)/2.0 # [0,1]
fixed_cond_data_normalized = 2*((tf.cast(fixed_cond_data_int, tf.float32)/255.)-.5) #normalized [-1,1]! 

init_op = tf.global_variables_initializer()  	# op to initialize the variables.

# Train loop
with tf.Session() as session:
    session.run(init_op)

    samples_255 = np.zeros((2*BATCH_SIZE, OUTPUT_DIM))
    real_flowimages = []
    for i in range(0, BATCH_SIZE):
        real_flowimg = [] # reset to be sure
        real_flowimg = fh.computeFlowImg(fixed_real_data[i].reshape((IM_DIM,IM_DIM,2)))  # (32, 32, 3) # now color img!! :)
        real_flowimage_T = np.transpose(real_flowimg, [2,0,1])  #  (3, 32, 32)
        real_flowimage = real_flowimage_T.reshape((OUTPUT_DIM,))  # instead of flatten? 
        real_flowimages.append(real_flowimage)
        print("real flowimages")
        print(real_flowimages.shape)
        

        real_flow = real_flowimage[i].astype('int32')
        print("real flow rgb now")
        print(real_flow.shape)
        samples_255[2*i+1,:] = real_flow # real flow color image 
        print("real flow inserted")  
        print(samples_255)
        last_frame = fixed_cond_data_int[i,OUTPUT_DIM:].astype('int32')   # need to transpose??
        print("last frame")
        print(last_frame.shape)
        samples_255[2*i,:] = last_frame # last frame left of generated sample
        print("last frame inserted")
        print(samples_255)
# samples_255= np.insert(samples_255, i*2, fixed_cond_data_int[i],axis=0)

    lib.save_images.save_images(samples_255.reshape((2*BATCH_SIZE, 3, IM_DIM, IM_DIM)), 'samples_{}.jpg'.format(frame)) # also save as .flo?
    real_flowims_np = np.asarray(real_flowimages, np.int32)
    print("real flowims")
    print(real_flowims_np.shape) 
    		

