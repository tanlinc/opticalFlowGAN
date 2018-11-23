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
#import tflib.SINTELdataFrame as sintel
import tflib.mnist as mnistloader
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

BATCH_SIZE = 50 # Batch size  ### for MNIST
HIDDEN_DIM = 64 # This overfits substantially; probably better 64; was 128
LAMBDA = 10 # Gradient penalty lambda hyperparameter
DISC_ITERS = 5 # How many discriminator iterations per generator iteration
ITERS = 100000 # #gen iters to train for; 200000 takes too long
IM_DIM = 28 # number of pixels along x and y (square assumed) ### for MNIST
OUTPUT_DIM = IM_DIM*IM_DIM # Number of pixels (3*32*32) ### for MNIST
CONTINUE = False  # Default False, set True if restoring from checkpoint
START_ITER = 0  # Default 0, set if restoring from checkpoint (100, 200,...)
CURRENT_PATH = "simpleGAN_MNIST"
restore_path = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/results/" + CURRENT_PATH + "/model.ckpt"  # change to Desktop

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

# initWeights
def linearStep(name, input_dim, output_dim, inputs):
    with tf.name_scope(name) as scope: 
        # initialization glorot
        weight_values = uniform(np.sqrt(2./(input_dim+output_dim)),(input_dim, output_dim))
        weight_values *= 1.0 # gain
        weight = lib.param(name + '.W', weight_values)   
        result = tf.matmul(inputs, weight) #inputs.get_shape().ndims == 2
        _biases = lib.param(name + '.b', np.zeros((output_dim,), dtype='float32'))
        result = tf.nn.bias_add(result, _biases)
    return result
    
def deconv2d(name, inputs, input_dim, output_dim, filter_size = 5, stride = 2):
    '''
    # inputs: tensor of shape (batch size, height, width, input_dim)
    # returns: tensor of shape (batch size, stride*height, stride*width, output_dim)
    '''
    with tf.name_scope(name) as scope:
        fan_in = input_dim * filter_size**2 / (stride**2) 
        fan_out = output_dim * filter_size**2             
        filters_stdev = np.sqrt(4./(fan_in+fan_out)) # he-init
        filter_values = uniform(filters_stdev, (filter_size, filter_size, output_dim, input_dim))
        filter_values *= 1.0 # * gain
        filters = lib.param(name+'.Filters', filter_values)
        inputs = tf.transpose(inputs, [0,2,3,1], name='NCHW_to_NHWC')
        input_shape = tf.shape(inputs)
        output_shape = tf.stack([input_shape[0], stride*input_shape[1], stride*input_shape[2], output_dim])
        result = tf.nn.conv2d_transpose(value=inputs, filter=filters, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')
        _biases = lib.param(name+'.Biases', np.zeros(output_dim, dtype='float32'))
        result = tf.nn.bias_add(result, _biases)
    return tf.transpose(result, [0,3,1,2], name='NHWC_to_NCHW')

def conv2d(name, inputs, input_dim, output_dim, filter_size = 5, stride = 2):
    with tf.name_scope(name) as scope:
        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)
        filters_stdev = np.sqrt(4./(fan_in+fan_out))
        filter_values = uniform(filters_stdev, (filter_size, filter_size, input_dim, output_dim))
        filter_values *= 1.0 # * gain
        filters = lib.param(name+'.Filters', filter_values)
        result = tf.nn.conv2d(input=inputs, filter=filters, strides=[1, 1, stride, stride], padding='SAME', data_format='NCHW')
        _biases = lib.param(name+'.Biases', np.zeros(output_dim, dtype='float32'))
        result = tf.nn.bias_add(result, _biases, data_format='NCHW')
    return result
    
def Generator(n_samples, noise=None):    
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    else:
        noise = tf.reshape(noise, [n_samples,128]) 

    output = linearStep('Generator.Input', 128, 4*4*4*HIDDEN_DIM, noise)

    #output = batchnorm([0], noise)    
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*HIDDEN_DIM, 4, 4])

    output = deconv2d('Generator.1', output, 4*HIDDEN_DIM, 2*HIDDEN_DIM, filter_size = 5)
    #output = batchnorm([0,2,3], output)
    output = tf.nn.relu(output)
    
    output = output[:,:,:7,:7]   ### for MNIST

    output = deconv2d('Generator.2', output, 2*HIDDEN_DIM, HIDDEN_DIM, filter_size = 5)
    #output = batchnorm([0,2,3], output)
    output = tf.nn.relu(output)
    
    output = deconv2d('Generator.3', output, HIDDEN_DIM, 1, filter_size = 5) ### for MNIST
    # output = tf.tanh(output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    inputs = tf.reshape(inputs, [-1, 1, IM_DIM, IM_DIM]) ### for MNIST

    output = conv2d('Discriminator.1', inputs, 1, HIDDEN_DIM, filter_size = 5) ### for MNIST
    output = LeakyReLU(output)

    output = conv2d('Discriminator.2', output, HIDDEN_DIM, 2*HIDDEN_DIM, filter_size = 5)
    output = LeakyReLU(output)

    output = conv2d('Discriminator.3', output, 2*HIDDEN_DIM, 4*HIDDEN_DIM, filter_size = 5)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*HIDDEN_DIM]) # adjust
    
    # last step to get single value
    output = linearStep('Discriminator.Output', 4*4*4*HIDDEN_DIM, 1, output)

    return tf.reshape(output, [-1])

#real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM]) ### for MNIST
#real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5) #[-1,1] ### for MNIST

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM]) ### for MNIST

fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

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
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

# Dataset iterators
#gen = sintel.load_train_gen(BATCH_SIZE, (IM_DIM,IM_DIM,3)) 
#dev_gen = sintel.load_test_gen(BATCH_SIZE, (IM_DIM,IM_DIM,3))

# Dataset iterator
train_gen, dev_gen, test_gen = mnistloader.load(BATCH_SIZE, BATCH_SIZE)	# load sets into global vars  ### for MNIST
def inf_train_gen():
    while True:			# iterator
        for images,targets in train_gen():
            yield images

gen = inf_train_gen()		# init iterator for training set ### for MNIST
fixed_data = next(gen) ### for MNIST  # (50, 784)
#samples_01 = ((samples+1.)/2.).astype('float32') # [0,1] ### for MNIST
#fixed_real_data_norm01 = tf.cast(fixed_data, tf.float32)/255. # [0,1] ### for MNIST

# For generating samples: define fixed noise
# fixed_samples, _ = next(gen)  # shape: (batchsize, 2*output_dim)
# fixed_real_data_int = fixed_samples[:,OUTPUT_DIM:]  # next frame as real for discr, shape (64,3072)
# fixed_real_data_norm01 = tf.cast(fixed_real_data_int, tf.float32)/255. # [0,1]
if(CONTINUE):
    fixed_noise = tf.get_variable("noise", shape=[BATCH_SIZE, 128]) # get noise from saved model
else:
    fixed_noise = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, 128]), name='noise') #var inst of const: var is saved, dtype float implicit
fixed_noise_samples = Generator(BATCH_SIZE, noise=fixed_noise) # Generator(n_samples,noise)

def generate_image(frame, true_dist):   # generates a batch of samples next to each other in one image!
    # test: generate some samples
    samples = session.run(fixed_noise_samples, feed_dict={real_data: fixed_data}) # [-1,1] ### for MNIST # (50, 784)
    samples_255 = ((samples+1.)*(255./2)).astype('int32') # [0,255]
    samples_01 = ((samples+1.)/2.).astype('float32') # [0,1]
    imsaver.save_images(samples_255.reshape((BATCH_SIZE, IM_DIM, IM_DIM)), 'samples_{}.png'.format(frame)) ### for MNIST
    print("Iteration %d : \n" % frame)
    # compare generated to real ones
    real = tf.reshape(fixed_data, [BATCH_SIZE,IM_DIM,IM_DIM,1]) ### for MNIST
    # real_gray = tf.image.rgb_to_grayscale(real) # tensor batch in&out returns original dtype = float [0,1] ### for MNIST
    pred = tf.reshape(samples_01, [BATCH_SIZE,IM_DIM,IM_DIM,1])  ### for MNIST
    # pred_gray = tf.image.rgb_to_grayscale(pred) ### for MNIST
    ssimval = tf.image.ssim(real, pred, max_val=1.0) # in tensor batch, out tensor ssimvals (64,)  ### for MNIST
    mseval_per_entry = tf.keras.metrics.mse(real, pred)  # mse on grayscale, on [0,1] ### for MNIST
    mseval = tf.reduce_mean(mseval_per_entry, [1,2])  ### for MNIST
    # ssimvals 0.2 to 0.75 :) # msevals 1-9 e -1 to -3
    ssimval_list = ssimval.eval()  # to numpy array # (64,)  # (50,)
    mseval_list = mseval.eval() # (64,)   # (50,)
    # print(ssimval_list)
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
         
    gen = inf_train_gen()		# init iterator for training set ### for MNIST

    for iteration in range(START_ITER, ITERS):  # START_ITER: 0 or from last checkpoint
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)
        # Train duscriminator
        for i in range(DISC_ITERS):
            _data  = next(gen)  # shape: (batchsize, 6144)  ### for MNIST
            #_real_data = _data[:,OUTPUT_DIM:] # current frame for disc ### for MNIST
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data: _data})# _real_data}) ### for MNIST
        plotter.plot('train disc cost', _disc_cost)
        plotter.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            _data = next(gen)  # shape: (batchsize, 6144) ### for MNIST
            #_real_data = _data[:,OUTPUT_DIM:] # current frame for disc ### for MNIST
            _dev_disc_cost = session.run(disc_cost, feed_dict={real_data: _data})# _real_data}) ### for MNIST
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