#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:06:09 2018

@author: linkermann

Wasserstein GAN simplified
"""
import os, sys
sys.path.append(os.getcwd())
#import tflib as lib
import tflib.save_images as imsaver
import tflib.plot as plotter
import tflib.mnist as mnistloader
import time
import numpy as np
import tensorflow as tf


BATCH_SIZE = 50 # Batch size needs to divide 50000
DIM = 64 # model dimensionality
LAMBDA = 10 # Gradient penalty lambda hyperparameter
DISC_ITERS = 5 # How many discriminator iterations per generator iteration
ITERS = 100000 # #gen iters to train for; 200000 takes too long
IM_DIM = 28 # number of pixels along x and y (square assumed) ### for MNIST
OUTPUT_DIM = IM_DIM*IM_DIM # Number of pixels (3*32*32) ### for MNIST
CONTINUE = False  # Default False, set True if restoring from checkpoint
START_ITER = 0  # Default 0, set if restoring from checkpoint (100, 200,...)
CURRENT_PATH = "simpleGAN_MNIST_2"
restore_path = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/results/" + CURRENT_PATH + "/model.ckpt"  # change to Desktop

_params = {}
      
def param(name, *args):
    """
    A wrapper for `tf.Variable` which enables parameter sharing.
    Creates and returns theano shared variables (`params`),
    so that you can easily search a graph for all params.
    """
    if name not in _params:
        var = tf.Variable(*args)
        _params[name] = var
    return _params[name]
        
def params_with_name(name):
    return [p for n,p in _params.items() if name in n]

def print_current_model_settings(locals_):
    all_vars = [(k,v) for (k,v) in locals_.items() if k.isupper()] 
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print("\t{}: {}".format(var_name, var_value))

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)
            
def uniform(stdev, size):
    return np.random.uniform(low=-stdev * np.sqrt(3), high=stdev * np.sqrt(3), size=size).astype('float32')

def init_weights_glorot(name, input_dim, output_dim):
    weight_values = uniform(np.sqrt(2./(input_dim+output_dim)),(input_dim, output_dim))
    weight = param(name+'.W', weight_values) 
    biases = param(name+'.b', np.zeros((output_dim,), dtype='float32'))
    return (weight, biases)

def linearStep(name, input_dim, output_dim, inputs):
    (weight, biases) = init_weights_glorot(name, input_dim, output_dim)
    result = tf.matmul(inputs, weight) #inputs.get_shape().ndims == 2
    result = tf.nn.bias_add(result, biases)
    return result

def init_deconv_filters_he(name, input_dim, output_dim, filter_size = 5, stride = 2): 
    fan_in = input_dim * filter_size**2 / (stride**2) 
    fan_out = output_dim * filter_size**2         
    filters_stdev = np.sqrt(4./(fan_in+fan_out))
    filter_values = uniform(filters_stdev, (filter_size, filter_size, output_dim, input_dim))
    filters = param(name+'.Filters', filter_values)
    biases = param(name+'.Biases', np.zeros(output_dim, dtype='float32'))
    return (filters, biases)
    
def deconv2d(name, inputs, input_dim, output_dim, filter_size = 5, stride = 2):
    '''
    # inputs: tensor of shape (batch size, height, width, input_dim)
    # returns: tensor of shape (batch size, stride*height, stride*width, output_dim)
    '''
    (filters, biases) = init_deconv_filters_he(name, input_dim, output_dim, filter_size = 5, stride = 2)
    inputs = tf.transpose(inputs, [0,2,3,1], name='NCHW_to_NHWC')
    input_shape = tf.shape(inputs)
    output_shape = tf.stack([input_shape[0], stride*input_shape[1], stride*input_shape[2], output_dim])
    result = tf.nn.conv2d_transpose(value=inputs, filter=filters, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')
    result = tf.nn.bias_add(result, biases)
    result = tf.transpose(result, [0,3,1,2], name='NHWC_to_NCHW')
    return result

def init_conv_filters_he(name, input_dim, output_dim, filter_size = 5, stride = 2): 
    fan_in = input_dim * filter_size**2 
    fan_out = output_dim * filter_size**2 / (stride**2)        
    filters_stdev = np.sqrt(4./(fan_in+fan_out))
    filter_values = uniform(filters_stdev, (filter_size, filter_size, input_dim, output_dim))
    filters = param(name+'.Filters', filter_values)
    biases = param(name+'.Biases', np.zeros(output_dim, dtype='float32'))
    return (filters, biases)

def conv2d(name, inputs, input_dim, output_dim, filter_size = 5, stride = 2):
    (filters, biases) = init_conv_filters_he(name, input_dim, output_dim, filter_size = 5, stride = 2)
    result = tf.nn.conv2d(input=inputs, filter=filters, strides=[1, 1, stride, stride], padding='SAME', data_format='NCHW')
    result = tf.nn.bias_add(result, biases, data_format='NCHW')
    return result
    
def Generator(n_samples, noise=None):    
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    else:
        noise = tf.reshape(noise, [n_samples,128]) 
    output = linearStep('Gen.Input', 128, 4*4*4*DIM, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])
    output = deconv2d('Gen.1', output, 4*DIM, 2*DIM, filter_size=5, stride=2)
    output = tf.nn.relu(output)
    output = output[:,:,:7,:7]  # because output needs to be 28x28
    output = deconv2d('Gen.2', output, 2*DIM, DIM, filter_size=5, stride=2)
    output = tf.nn.relu(output)
    output = deconv2d('Gen.3', output, DIM, 1, filter_size=5, stride=2)
    output = tf.nn.sigmoid(output) # sigmoid instead of tanh for mnist
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    inputs = tf.reshape(inputs, [-1, 1, IM_DIM, IM_DIM])
    output = conv2d('Disc.1', inputs, 1, DIM, filter_size=5, stride=2)
    output = LeakyReLU(output)
    output = conv2d('Disc.2', output, DIM, 2*DIM, filter_size=5, stride=2)
    output = LeakyReLU(output)
    output = conv2d('Disc.3', output, 2*DIM, 4*DIM, filter_size=5, stride=2)
    output = LeakyReLU(output)
    output = tf.reshape(output, [-1, 4*4*4*DIM]) # adjust 
    output = linearStep('Disc.Output', 4*4*4*DIM, 1, output) # to single value
    return tf.reshape(output, [-1])

#------------------------------------------------------------------------------

print_current_model_settings(locals().copy())

if(CONTINUE):
    tf.reset_default_graph()
    
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM]) 
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = params_with_name('Gen')
disc_params = params_with_name('Disc')

# WGAN-GP ---------------------------------------------------------------
# Standard WGAN loss
gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

# Gradient penalty on Discriminator 
alpha = tf.random_uniform(shape=[BATCH_SIZE,1], minval=0., maxval=1.)
interpolates = real_data + (alpha*(fake_data - real_data))
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
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

if(CONTINUE):
    fixed_noise = tf.get_variable("noise", shape=[BATCH_SIZE, 128]) # get noise from saved model
else:
    fixed_noise = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, 128]), name='noise') #var inst of const: var is saved, dtype float implicit
    # fixed_noise.trainable = False
fixed_noise_samples = Generator(BATCH_SIZE, noise=fixed_noise) # Generator(n_samples,noise)

def generate_image(frame, true_dist):   # test: generates a batch of samples next to each other in one image!
    samples = session.run(fixed_noise_samples) # [-1,1] 
    samples_255 = ((samples+1.)*(255./2)).astype('int32') # [0,255]
    imsaver.save_images(samples_255.reshape((BATCH_SIZE, IM_DIM, IM_DIM)), 'samples_{}.png'.format(frame))   
        
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
    
    gen = inf_train_gen()		# init iterator for training set
    dev_generator = inf_dev_gen()		# init iterator for training set 

    for iteration in range(START_ITER, ITERS):  # START_ITER: 0 or from last checkpoint
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)
        # Train duscriminator
        for i in range(DISC_ITERS):
            _data  = next(gen) 
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data: _data})
        plotter.plot('train disc cost', _disc_cost)
        plotter.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            _data = next(dev_generator) 
            _dev_disc_cost = session.run(disc_cost, feed_dict={real_data: _data})
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