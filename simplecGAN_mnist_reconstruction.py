#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:06:09 2018

@author: linkermann

Wasserstein GAN simplified
"""
import os, sys
sys.path.append(os.getcwd())
import tflib.save_images as imsaver
import tflib.plot as plotter
import tflib.mnist as mnistloader
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as lays

BATCH_SIZE = 50 # Batch size needs to divide 50000
DIM = 64 # model dimensionality
LAMBDA = 10 # Gradient penalty lambda hyperparameter
DISC_ITERS = 5 # How many discriminator iterations per generator iteration
ITERS = 100000 # How many generator iterations to train for # 200000 takes too long
IM_DIM = 28 # number of pixels along x and y (square assumed)
OUTPUT_DIM = IM_DIM*IM_DIM # Number of pixels (28*28)
CONTINUE = False  # Default False, set True if restoring from checkpoint
START_ITER = 0  # Default 0, set accordingly if restoring from checkpoint (100, 200, ...)
CURRENT_PATH = "simplecGAN_MNIST_reconstruction_refactored"
restore_path = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/results/" + CURRENT_PATH + "/model.ckpt" # changed for desktop
 

def print_current_model_settings(locals_):
    all_vars = [(k,v) for (k,v) in locals_.items() if k.isupper()] 
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print("\t{}: {}".format(var_name, var_value))

def Generator(n_samples, conditions, noise=None):    
    if noise is None:
        noise = tf.random_normal([n_samples, IM_DIM*IM_DIM])
    noise = tf.reshape(noise, [n_samples, 1, IM_DIM, IM_DIM])
    # conditional input: last frame
    conds = tf.reshape(conditions, [n_samples, 1, IM_DIM, IM_DIM])  # (50,...) TO (50,1,28,28)
    # for now just concat the inputs: noise as second dim of cond image 
    output = tf.concat([noise, conds], 1)  # to: (BATCH_SIZE,2,28,28)
    output = tf.reshape(output, [n_samples, IM_DIM*IM_DIM*2]) # 28x28x2 = 784x2; to: (50, ...)
    output = lays.fully_connected(output, 4*4*4*DIM, # expand noise input
                     weights_initializer=tf.initializers.glorot_uniform(), 
                     reuse = tf.AUTO_REUSE, scope = 'Gen.Input')
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])
    output = tf.transpose(output, [0,2,3,1], name='NCHW_to_NHWC')
    output = lays.conv2d_transpose(output, 2*DIM, kernel_size= 5, stride = 2, 
            weights_initializer=tf.initializers.he_uniform(),
            reuse=tf.AUTO_REUSE, scope='Gen.1')
    output = output[:,:7,:7,:]  # because output needs to be 28x28
    output = lays.conv2d_transpose(output, DIM, kernel_size = 5, stride = 2, 
            weights_initializer=tf.initializers.he_uniform(),
            reuse=tf.AUTO_REUSE, scope='Gen.2')
    output = lays.conv2d_transpose(output, 1, kernel_size = 5, stride = 2, 
            activation_fn=tf.nn.sigmoid, #tf.tanh,  #tf.nn.sigmoid
            weights_initializer=tf.initializers.he_uniform(),
            reuse=tf.AUTO_REUSE, scope='Gen.3')
    output = tf.transpose(output, [0,3,1,2], name='NHWC_to_NCHW')
    return tf.layers.Flatten()(output) #tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, conditions):
    inputs = tf.reshape(inputs, [-1, 1, IM_DIM, IM_DIM]) 
    # conditional input: last frame
    conds = tf.reshape(conditions, [-1, 1, IM_DIM, IM_DIM])     
    # for now just concat the inputs
    ins = tf.concat([inputs, conds], 1) #to: (BATCH_SIZE, 2, 28, 28)
    output = lays.conv2d(ins, DIM, kernel_size = 5, stride = 2, 
            data_format='NCHW',
            activation_fn=tf.nn.leaky_relu, 
            weights_initializer=tf.initializers.he_uniform(),
            reuse=tf.AUTO_REUSE, scope='Disc.1')
    output = lays.conv2d(output, 2*DIM, kernel_size = 5, stride = 2, 
            data_format='NCHW',
            activation_fn=tf.nn.leaky_relu, 
            weights_initializer=tf.initializers.he_uniform(),
            reuse=tf.AUTO_REUSE, scope='Disc.2')
    output = lays.conv2d(output, 4*DIM, kernel_size = 5, stride = 2,
            data_format='NCHW',
            activation_fn=tf.nn.leaky_relu, 
            weights_initializer=tf.initializers.he_uniform(),
            reuse=tf.AUTO_REUSE, scope='Disc.3')
    output = tf.reshape(output, [-1, 4*4*4*DIM]) # adjust
    output = lays.fully_connected(output, 1, activation_fn=None, # to single value
                     weights_initializer=tf.initializers.glorot_uniform(), 
                     reuse = tf.AUTO_REUSE, scope = 'Disc.Output')
    return tf.reshape(output, [-1])

#-----------------------------------------------------------------------------

print_current_model_settings(locals().copy())

#if(CONTINUE):
tf.reset_default_graph()

cond_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM]) # conditional input for both G and D
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM]) 
fake_data = Generator(BATCH_SIZE, cond_data)

disc_real = Discriminator(real_data, cond_data)
disc_fake = Discriminator(fake_data, cond_data)

gen_params = [var for var in tf.get_collection("variables") if "Gen" in var.name]
disc_params = [var for var in tf.get_collection("variables") if "Disc" in var.name]

# WGAN-GP ---------------------------------------------------------------
# Standard WGAN loss
gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

# Gradient penalty on Discriminator 
alpha = tf.random_uniform(shape=[BATCH_SIZE,1], minval=0., maxval=1.)
interpolates = real_data + (alpha*(fake_data - real_data))
gradients = tf.gradients(Discriminator(interpolates, cond_data), [interpolates])[0] #added cond here
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

# Dataset iterator
train_gen, dev_gen, test_gen = mnistloader.load(BATCH_SIZE, BATCH_SIZE)	# load sets into global vars
def inf_train_gen():
    while True:			# iterator
        for images,targets in train_gen():
            yield images
            
def inf_dev_gen():
    while True:			# iterator
        for images,targets in dev_gen():
            yield images
            
def inf_test_gen():
    while True:			# iterator
        for images,targets in test_gen():
            yield images

gen = inf_train_gen()		# init iterator for training set
dev_generator = inf_dev_gen()		# init iterator for training set 
test_generator = inf_test_gen()		# init iterator for training set 

# For generating samples: define fixed noise and conditional input
fixed_cond_data = next(test_generator) # fixed_real_data
fixed_cond_data_255 = ((fixed_cond_data)*255.).astype('int32') # [0,255] 

if(CONTINUE):
    fixed_noise = tf.get_variable("noise", shape=[BATCH_SIZE, IM_DIM*IM_DIM]) # get noise from saved model
else:
    fixed_noise = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, IM_DIM*IM_DIM]), name='noise') #var inst of const: var is saved, dtype float implicit  # shape for additional channel
fixed_noise_samples = Generator(BATCH_SIZE, fixed_cond_data, noise=fixed_noise) # Generator(n_samples,conds, noise)

def generate_image(frame, true_dist):   # generates a batch of samples next to each other in one image!
    samples = session.run(fixed_noise_samples, feed_dict={real_data: fixed_cond_data, cond_data: fixed_cond_data}) # [-1,1]
    samples_255 = ((samples+1.)*(255./2)).astype('int32') # [0,255] 
    samples_01 = ((samples+1.)/2.).astype('float32') # [0,1]
    for i in range(0, BATCH_SIZE):
        samples_255 = np.insert(samples_255, i*2, fixed_cond_data_255[i,:], axis=0) # show cond digit next to generated sample
    imsaver.save_images(samples_255.reshape((2*BATCH_SIZE, 1, IM_DIM, IM_DIM)), 'samples_{}.jpg'.format(frame))
    print("Iteration %d : \n" % frame)
    # compare generated to real one
    real = tf.reshape(fixed_cond_data, [BATCH_SIZE,IM_DIM,IM_DIM,1])
    pred = tf.reshape(samples_01, [BATCH_SIZE,IM_DIM,IM_DIM,1])
    ssimval = tf.image.ssim(real, pred, max_val=1.0) # tensor batch in, out tensor of ssimvals (64,)
    mseval_per_entry = tf.keras.metrics.mse(real, pred)  # mse on grayscale, on [0,1]
    mseval = tf.reduce_mean(mseval_per_entry, [1,2])
    ssimval_list = ssimval.eval()  # to numpy array # (50,)
    mseval_list = mseval.eval() # (50,)
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
         plotter.restore(START_ITER)  # makes plots start from 0
    else:
         session.run(init_op)	

    for iteration in range(START_ITER, ITERS):  # START_ITER: 0 or from last checkpoint
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _data = next(gen)  # shape: (batchsize, 6144)
            _cond_data = _data # digit as cond
            _ = session.run(gen_train_op, feed_dict={cond_data: _cond_data})
        # Train duscriminator
        for i in range(DISC_ITERS):
            _data = next(gen)  # shape: (batchsize, 6144)
            _cond_data = _data # digit as cond
            _real_data = _data # current frame for disc
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data: _real_data, cond_data: _cond_data})
        plotter.plot('train disc cost', _disc_cost)
        plotter.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            _data = next(dev_generator)  # shape: (batchsize, 6144)
            _cond_data = _data # digit as cond
            _real_data = _data # current frame for disc
            _dev_disc_cost = session.run(disc_cost, feed_dict={real_data: _real_data, cond_data: _cond_data})
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