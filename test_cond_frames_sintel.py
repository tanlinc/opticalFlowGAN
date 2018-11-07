import os, sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
import tflib as lib
import tflib.save_images
import tflib.flow_handler as fh
import tflib.SINTELdata as sintel

outpath = "/home/linkermann/opticalFlow/opticalFlowGAN/data/gentest/"

BATCH_SIZE = 64 
IM_DIM = 32 # number of pixels along x and y (square assumed)
OUTPUT_DIM = IM_DIM*IM_DIM*3 # Number of pixels (3*32*32) - rgb color
OUTPUT_DIM_FLOW = IM_DIM*IM_DIM*2 # Number of pixels (2*32*32) - uv
lib.print_model_settings(locals().copy())

# Dataset iterator: # batch size, im size, im size flow
gen = sintel.load_train_gen(BATCH_SIZE, (IM_DIM,IM_DIM,3), (IM_DIM,IM_DIM,2)) 
_data, _flow  = next(gen) # fixed_cond_samples, fixed_flow_samples # (batchsize, 3072) 
fixed_cond_data_int = _data[:,0:2*OUTPUT_DIM] # earlier frames as cond, _data: (64,3*3072)
fixed_viz_data_int = _data[:,OUTPUT_DIM:2*OUTPUT_DIM] # each later frame for viz
fixed_real_data =_flow[:,OUTPUT_DIM_FLOW:] # later flow for discr, _flow: (64,2*2048)
print(fixed_real_data.shape)
images = fixed_viz_data_int.reshape(BATCH_SIZE,3,IM_DIM,IM_DIM)
# tflib.save_images.save_images(images, outpath+"condvizbatch.jpg") # viz cond data

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(0, BATCH_SIZE):
        real_uvflow = fixed_real_data[i,:]
        real_uvflow = real_uvflow.reshape((IM_DIM,IM_DIM,2))
        real_flowimg = fh.computeFlowImg(real_uvflow)  # (32, 32, 3) color img!
        real_flowimg = real_flowimg.reshape(IM_DIM,IM_DIM,3).astype('int32') # (32, 32, 3) 
        real_flowimg_T = np.transpose(real_flowimg, [2,0,1])  #  (3, 32, 32) 
        images = np.insert(images, i*2+1, real_flowimg_T, axis=0)
    # images.shape: (128, 3, 32, 32) = (2*BATCH_SIZE, 3, IM_DIM, IM_DIM)
    lib.save_images.save_images(images, 'cond_frames_batch.jpg')    		

flows = _flow[0] # first from batch
flow1 = flows[0:OUTPUT_DIM_FLOW] # (2048,) for 32
flow1 = flow1.reshape(IM_DIM,IM_DIM,2)
flowimg1 = fh.computeFlowImg(flow1)    # (200, 200, 3) # now color img!! :)
flowimg1 = flowimg1.reshape(IM_DIM,IM_DIM,3)
flowimg1_T = np.transpose(flowimg1, [2,0,1])  #  (3, 200, 200)
lib.save_images.save_images(flowimg1_T.reshape((1,3,IM_DIM,IM_DIM)), outpath+"realflowsamplesintel1.jpg")

flow2 = flows[OUTPUT_DIM_FLOW:] # (2048,) for 32
flow2 = flow2.reshape(IM_DIM,IM_DIM,2)
flowimg2 = fh.computeFlowImg(flow2)    # (200, 200, 3) # now color img!! :)
flowimg2 = flowimg2.reshape(IM_DIM,IM_DIM,3)
# flowimg2_T = np.transpose(flowimg2, [2,0,1])  #  (3, 200, 200)
lib.save_images.save_images(flowimg2_T.reshape((1,3,IM_DIM,IM_DIM)), outpath+"realflowsamplesintel2.jpg")
