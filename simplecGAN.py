#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:06:09 2018

@author: linkermann

Wasserstein GAN simplified
"""
import os, sys
sys.path.append(os.getcwd())
import tflib as lib
import tflib.save_images as imsaver
import tflib.plot as plotter
import time
import numpy as np
import tensorflow as tf
import tflib.SINTELdataFrame as sintel
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt


HIDDEN_DIM = 64 # This overfits substantially; you're probably better off with 64 # or 128?
LAMBDA = 10 # Gradient penalty lambda hyperparameter
DISC_ITERS = 5 # How many discriminator iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 100000 # How many generator iterations to train for # 200000 takes too long
IM_DIM = 32 # number of pixels along x and y (square assumed)
SQUARE_IM_DIM = IM_DIM*IM_DIM # 32*32 = 1024
OUTPUT_DIM = SQUARE_IM_DIM*3 # Number of pixels (3*32*32)
CONTINUE = True  # Default False, set True if restoring from checkpoint
START_ITER = 600  # Default 0, set accordingly if restoring from checkpoint (100, 200, ...)
CURRENT_PATH = "sintel/simpleGAN"
restore_path = "/home/linkermann/opticalFlow/opticalFlowGAN/results/" + CURRENT_PATH + "/model.ckpt"
 
lib.print_model_settings(locals().copy())

if(CONTINUE):
    tf.reset_default_graph()

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)
            
def uniform(stdev, size):
    return np.random.uniform(low=-stdev * np.sqrt(3), high=stdev * np.sqrt(3), size=size).astype('float32')

#def batchnorm(axes, inputs):
#    if axes == [0]:
#        mean, var = tf.nn.moments(inputs, [0], keep_dims=True)
#        shape = mean.get_shape().as_list()
#        offset = np.zeros(shape, dtype='float32')
#        scale = np.ones(shape, dtype='float32')
#        return tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)
#        # or tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-2, mean=moving_mean, variance=moving_variance, is_training=False, data_format='NCHW')
#    else: # axes = [0,2,3]    
#        inputs = tf.transpose(inputs, [0,2,3,1])
#        mean, var = tf.nn.moments(inputs, [0,1,2], keep_dims=False)
#        offset = np.zeros(mean.get_shape()[-1], dtype='float32')
#        scale = np.ones(var.get_shape()[-1], dtype='float32')
#        result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-4)
#        return tf.transpose(result, [0,3,1,2])
    
def deconv2d(inputs, input_dim, output_dim, filter_size = 5, stride = 2):
    '''
    # inputs: tensor of shape (batch size, height, width, input_dim)
    # returns: tensor of shape (batch size, stride*height, stride*width, output_dim)
    '''
    fan_in = input_dim * filter_size**2 / (stride**2) 
    fan_out = output_dim * filter_size**2             
    filters_stdev = np.sqrt(4./(fan_in+fan_out)) # he-init
    filter_values = uniform(filters_stdev, (filter_size, filter_size, output_dim, input_dim))
    filter_values *= 1.0 # * gain
    #filters = lib.param(name+'.Filters', filter_values)
    inputs = tf.transpose(inputs, [0,2,3,1], name='NCHW_to_NHWC') # input is output tensor
    input_shape = tf.shape(inputs)
    output_shape = tf.pack([input_shape[0], stride*input_shape[1], stride*input_shape[2], output_dim])
    result = tf.nn.conv2d_transpose(value=inputs, filter=filter_values, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')
    #_biases = lib.param(name+'.Biases', np.zeros(output_dim, dtype='float32'))
    # result = tf.nn.bias_add(result, _biases)
    return tf.transpose(result, [0,3,1,2], name='NHWC_to_NCHW')

def conv2d(inputs, input_dim, output_dim, filter_size = 5, stride = 2):
    # name as param; with tf.name_scope(name) as scope:
    fan_in = input_dim * filter_size**2
    fan_out = output_dim * filter_size**2 / (stride**2)
    filters_stdev = np.sqrt(4./(fan_in+fan_out))
    filter_values = uniform(filters_stdev, filter_size, filter_size, input_dim, output_dim)
    filter_values *= 1.0 # * gain
    # filters = lib.param(name+'.Filters', filter_values)
    result = tf.nn.conv2d(input=inputs, filter=filter_values, strides=[1, 1, stride, stride], padding='SAME', data_format='NCHW')
    #_biases = lib.param(name+'.Biases', np.zeros(output_dim, dtype='float32'))
    #result = tf.nn.bias_add(result, _biases, data_format='NCHW')
    return result
    
def Generator(n_samples, conditions, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, SQUARE_IM_DIM])
    noise = tf.reshape(noise, [n_samples, 1, IM_DIM, IM_DIM])
    
    # conditional input: last frame
    conds = tf.reshape(conditions, [n_samples, 3, IM_DIM, IM_DIM])  # (64,3072) TO (64,3,32,32)

    # for now just concat the inputs: noise as fourth dim of cond image 
    output = tf.concat([noise, conds], 1)  # to: (BATCH_SIZE,4,32,32)
    output = tf.reshape(output, [n_samples, SQUARE_IM_DIM*4]) # 32x32x4 = 4096; to: (BATCH_SIZE, 4096)

    # --> orth initialization? (linear)

    #output = batchnorm([0], output)    
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*HIDDEN_DIM, 4, 4])

    output = deconv2d(output, 4*HIDDEN_DIM, 2*HIDDEN_DIM, filter_size = 5)
    #output = batchnorm([0,2,3], output)
    output = tf.nn.relu(output)

    output = deconv2d(output, 2*HIDDEN_DIM, HIDDEN_DIM, filter_size = 5)
    #output = batchnorm([0,2,3], output)
    output = tf.nn.relu(output)
    
    output = deconv2d(output, HIDDEN_DIM, 3, filter_size = 5)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, conditions):
    inputs = tf.reshape(inputs, [-1, 3, IM_DIM, IM_DIM])
    
    # conditional input: last frame
    conds = tf.reshape(conditions, [-1, 3, IM_DIM, IM_DIM])  
    
    # for now just concat the inputs
    ins = tf.concat([inputs, conds], 1) #to: (BATCH_SIZE, 6, 32, 32)

    output = conv2d(ins, 6, HIDDEN_DIM, filter_size = 5)
    output = LeakyReLU(output)

    output = conv2d(output, HIDDEN_DIM, 2*HIDDEN_DIM, filter_size = 5)
    output = LeakyReLU(output)

    output = conv2d(output, 2*HIDDEN_DIM, 4*HIDDEN_DIM, filter_size = 5)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*8*HIDDEN_DIM]) # adjust
    
    #last step to get one value out!!!!!!!!!!
    # output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*DIM, 1, output) # 4 inst of 8??

    return tf.reshape(output, [-1])

cond_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM]) # conditional input for both G and D
cond_data = 2*((tf.cast(cond_data_int, tf.float32)/255.)-.5) #normalized [-1,1]

real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5) #normalized [-1,1]

fake_data = Generator(BATCH_SIZE, cond_data)

disc_real = Discriminator(real_data, cond_data)
disc_fake = Discriminator(fake_data, cond_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

# WGAN-GP ---------------------------------------------------------------
# Standard WGAN loss
gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

# Gradient penalty on Discriminator 
alpha = tf.random_uniform(shape=[BATCH_SIZE,1], minval=0., maxval=1.)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates, cond_data), [interpolates])[0] #added cond here
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

# Dataset iterators
gen = sintel.load_train_gen(BATCH_SIZE, (IM_DIM,IM_DIM,3)) 
dev_gen = sintel.load_test_gen(BATCH_SIZE, (IM_DIM,IM_DIM,3))

# For generating samples: define fixed noise and conditional input
fixed_cond_samples, _ = next(gen)  # shape: (batchsize, 2*output_dim)
fixed_cond_data_int = fixed_cond_samples[:,0:OUTPUT_DIM]  # last frame as cond, shape (64,3072)
fixed_real_data_int = fixed_cond_samples[:,OUTPUT_DIM:]  # next frame as real for discr, shape (64,3072)
fixed_cond_data_norm = 2*((tf.cast(fixed_cond_data_int, tf.float32)/255.)-.5) # [-1,1]
fixed_real_data_norm01 = tf.cast(fixed_real_data_int, tf.float32)/255. # [0,1]
if(CONTINUE):
    fixed_noise = tf.get_variable("noise", shape=[BATCH_SIZE, SQUARE_IM_DIM]) # get noise from saved model
else:
    fixed_noise = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, SQUARE_IM_DIM]), name='noise') #var inst of const: var is saved, dtype float implicit  # shape for additional channel
fixed_noise_samples = Generator(BATCH_SIZE, fixed_cond_data_norm, noise=fixed_noise) # Generator(n_samples,conds, noise)

def generate_image(frame, true_dist):   # generates a batch of samples next to each other in one image!
    samples = session.run(fixed_noise_samples, feed_dict={real_data_int: fixed_real_data_int, cond_data_int: fixed_cond_data_int}) # [-1,1]
    samples_255 = ((samples+1.)*(255./2)).astype('int32') # [0,255] 
    samples_01 = ((samples+1.)/2.).astype('float32') # [0,1]
    for i in range(0, BATCH_SIZE):
        samples_255= np.insert(samples_255, i*2, fixed_cond_data_int[i],axis=0) # show last frame next to generated sample
    imsaver.save_images(samples_255.reshape((2*BATCH_SIZE, 3, IM_DIM, IM_DIM)), 'samples_{}.jpg'.format(frame))
    print("Iteration %d : \n" % frame)
    # compare generated to real one
    real = tf.reshape(fixed_real_data_norm01, [BATCH_SIZE,IM_DIM,IM_DIM,3])
    real_gray = tf.image.rgb_to_grayscale(real) # tensor batch in&out; returns original dtype = float [0,1]
    pred = tf.reshape(samples_01, [BATCH_SIZE,IM_DIM,IM_DIM,3])
    pred_gray = tf.image.rgb_to_grayscale(pred)
    ssimval = tf.image.ssim(real_gray, pred_gray, max_val=1.0) # tensor batch in, out tensor of ssimvals (64,)
    mseval_per_entry = tf.keras.metrics.mse(real_gray, pred_gray)  # mse on grayscale, on [0,1]
    mseval = tf.reduce_mean(mseval_per_entry, [1,2])
    # ssimvals 0.2 to 0.75 :) # msevals 1-9 e -1 to -3
    ssimval_list = ssimval.eval()  # to numpy array # (64,)
    mseval_list = mseval.eval() # (64,)
    #print(ssimval_list)
    # print(mseval_list)
    for i in range (0,3):
        plotter.plot('SSIM for sample %d' % (i+1), ssimval_list[i])
        plotter.plot('MSE for sample %d' % (i+1), mseval_list[i])
        print("sample %d \t MSE: %.5f \t SSIM: %.5f \r\n" % (i, mseval_list[i], ssimval_list[i]))
        
        
init_op = tf.global_variables_initializer()  	# op to initialize the variables.
saver = tf.train.Saver()			# ops to save and restore all the variables.

# Train loop
with tf.Session() as session:
    # Init variables
    if(CONTINUE):
         saver.restore(session, restore_path) # Restore variables from saved session.
         print("Model restored.")
         plotter.restore(START_ITER)  # does not fully work, but makes plots start from newly started iteration
    else:
         session.run(init_op)	

    for iteration in range(START_ITER, ITERS):  # START_ITER: 0 or from last checkpoint
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _data, _ = next(gen)  # shape: (batchsize, 6144)
            _cond_data = _data[:,0:OUTPUT_DIM] # last frame as cond
            _ = session.run(gen_train_op, feed_dict={cond_data_int: _cond_data})
        # Train duscriminator
        for i in range(DISC_ITERS):
            _data, _ = next(gen)  # shape: (batchsize, 6144)
            _cond_data = _data[:,0:OUTPUT_DIM] # last frame as cond
            _real_data = _data[:,OUTPUT_DIM:] # current frame for disc
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_int: _real_data, cond_data_int: _cond_data})
        plotter.plot('train disc cost', _disc_cost)
        plotter.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            _data, _ = next(gen)  # shape: (batchsize, 6144)
            _cond_data = _data[:,0:OUTPUT_DIM] # last frame as cond
            _real_data = _data[:,OUTPUT_DIM:] # current frame for disc
            _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: _real_data, cond_data_int: _cond_data})
            dev_disc_costs.append(_dev_disc_cost)
            plotter.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, _data)
            save_path = saver.save(session, restore_path) # Save the variables to disk.
            print("Model saved in path: %s" % save_path)
            # chkp.print_tensors_in_checkpoint_file("model.ckpt", tensor_name='', all_tensors=True)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            plotter.flush()

        plotter.tick()