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
#import tflib.mnist as mnistloader
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as lays
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt 

EXPERIMENT = "mnist_useLabel_likeCGANMaster" #results/

BATCH_SIZE = 50 # Batch size needs to divide 50000 # 64?
DIM = 64 # model dimensionality
LAMBDA = 10 # Gradient penalty lambda hyperparameter
DISC_ITERS = 5 # How many discriminator iterations per generator iteration
ITERS = 10000 # How many generator iterations to train for # was 200000
IM_DIM = 28 # number of pixels along x and y (square assumed) for generated image
OUTPUT_DIM = IM_DIM*IM_DIM # Number of pixels (28*28)
CONTINUE = False  # Default False, set True if restoring from checkpoint
START_ITER = 0  # Default 0, set accordingly if restoring from checkpoint (100, 200, ...)

restore_path = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/" + EXPERIMENT + "/model/model.ckpt" # the path of model, changed for desktop
samples_path = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/" + EXPERIMENT + "/samples/" # changed for desktop
# visua_path = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/" + CURRENT_PATH + "/visualization/" # changed for desktop

LEARNING_RATE = 1e-4 # 0.0002 # the learning rate for gan
Z_DIM = 100 # the dimension of noise z
COND_DIM = 10 # the dimension of condition y
log_dir = "/tmp/tensorflow_mnist" # the path of tensorflow's log
TRAIN_TEST_VIZ = 0 # 0: train ; 1:test ; 2:visualize

# Start tensorboard in the current folder:
#  tensorboard --logdir=logdir
# Open 'Webbrowser at http://localhost:6006/
 
def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims

def print_current_model_settings(locals_):
    all_vars = [(k,v) for (k,v) in locals_.items() if k.isupper()] 
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print("\t{}: {}".format(var_name, var_value))
    
def Generator(n_samples,conditions, noise=None): 
    # with tf.variable_scope('generator') as scope:   
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    yb = tf.reshape(conditions, shape=[BATCH_SIZE, 1, 1, COND_DIM])
    z = tf.concat([noise, labels], 1)  # to: (BATCH_SIZE,128+10)
    output = lays.fully_connected(z, 4*4*4*DIM, # expand noise input  # to 1024
                     weights_initializer=tf.initializers.glorot_uniform(), 
                     reuse = tf.AUTO_REUSE, scope = 'Gen.Input')  #  batch_normal in there as well..
    d1 = tf.concat([output, labels], 1)
    output = lays.fully_connected(d1, 7*7*2*64, # expand noise input  # to 1024
                     weights_initializer=tf.initializers.glorot_uniform(), 
                     reuse = tf.AUTO_REUSE, scope = 'Gen.fully') #  batch_normal in there as well..
    d2 = tf.reshape(output, [BATCH_SIZE, 7, 7, 64 * 2])
    d2 = conv_cond_concat(d2, yb)
    # output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = tf.transpose(d2, [0,2,3,1], name='NCHW_to_NHWC')
    output = lays.conv2d_transpose(output, 2*DIM, kernel_size= 5, stride = 2,  # to 128: [batch_size, 14, 14, 128]
            weights_initializer=tf.initializers.he_uniform(),
            reuse=tf.AUTO_REUSE, scope='Gen.deconv1')  # include batch_norm

    d3 = conv_cond_concat(d3, yb)

    output = lays.conv2d_transpose(d3, 1, kernel_size = 5, stride = 2,  # [batch_size, im_dim, im_dim, channel]
            activation_fn=tf.nn.sigmoid,
            weights_initializer=tf.initializers.he_uniform(), # xavier_initializer()
            reuse=tf.AUTO_REUSE, scope='Gen.deconv2')
    output = tf.transpose(output, [0,3,1,2], name='NHWC_to_NCHW')
    return tf.layers.Flatten()(output) #tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, conditions): # reuse=False
    # with tf.variable_scope("discriminator") as scope:  ## if reuse == True: scope.reuse_variables()
    inputs = tf.reshape(inputs, [-1, IM_DIM, IM_DIM, 1]) 
    yb = tf.reshape(conditions, shape=[BATCH_SIZE, 1, 1, COND_DIM])
    concat_data = conv_cond_concat(inputs, yb)
    # output = tf.transpose(inputs, [0,2,3,1], name='NCHW_to_NHWC')
    conv1 = lays.conv2d(concat_data, 10, kernel_size = 5, stride = 2, # 10 = DIM
            #data_format='NCHW', 
            activation_fn=tf.nn.leaky_relu, 
            weights_initializer=tf.initializers.he_uniform(),
            reuse=tf.AUTO_REUSE, scope='Disc.conv1')
    conv1 = conv_cond_concat(conv1, yb)
    conv2 = lays.conv2d(conv1, 64, kernel_size = 5, stride = 2, # 64 = 2*DIM
            #data_format='NCHW', 
            activation_fn=tf.nn.leaky_relu, 
            weights_initializer=tf.initializers.he_uniform(),
            reuse=tf.AUTO_REUSE, scope='Disc.conv2') # batch norm
    # output = tf.transpose(output, [0,3,1,2], name='NHWC_to_NCHW')
    output = tf.reshape(conv2, [BATCH_SIZE, -1]) # adjust  ##  [-1, 4*4*4*DIM]
    output = tf.concat([output, conditions], 1)
    output = lays.fully_connected(output, 1024, activation_fn=tf.nn.leaky_relu, 
                     weights_initializer=tf.initializers.glorot_uniform(), 
                     reuse = tf.AUTO_REUSE, scope = 'Disc.fully') # batch norm
    f1 = tf.concat([output, conditions], 1)
    output = lays.fully_connected(f1, 1, activation_fn=None, # to single value
                     weights_initializer=tf.initializers.glorot_uniform(),  # xavier_initializer()
                     reuse = tf.AUTO_REUSE, scope = 'Disc.Output')
    out = tf.reshape(output, [-1])
    return tf.nn.sigmoid(out), out

#-----------------------------------------------------------------------------

print_current_model_settings(locals().copy())

#if(CONTINUE):
tf.reset_default_graph()

condition_data = tf.placeholder(tf.int32, shape=[BATCH_SIZE]) # conditional input for both G and D

# z = tf.placeholder(tf.float32, [BATCH_SIZE, Z_DIM])
y = tf.placeholder(tf.float32, [BATCH_SIZE, COND_DIM])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM]) # call images
fake_data = Generator(BATCH_SIZE, condition_data) # give z here?
# G_image = tf.summary.image("G_out", fake_data) # already in  Generator!
#for _,gen_im in fake_data: # add a histogram for the gradient
#    tf.summary.histogram("{}-generated".format(gen_im.name.replace(":","_")), gen_im)

disc_real = Discriminator(real_data, condition_data)
d_real_out = tf.summary.tensor_summary("discriminator real output",disc_real)
disc_fake = Discriminator(fake_data, condition_data)
d_fake_out = tf.summary.tensor_summary("discriminator fake output",disc_fake)

#D_pro, D_logits = Discriminator(real_data, y, False)
#D_pro_sum = tf.summary.histogram("D_pro", D_pro)
#G_pro, G_logits = Discriminator(fake_data, y, True)
#G_pro_sum = tf.summary.histogram("G_pro", G_pro)

t_vars = tf.trainable_variables() # or tf.get_collection("variables")
gen_params = [var for var in t_vars if "Gen" in var.name]
disc_params = [var for var in t_vars if "Disc" in var.name]

# WGAN-GP ---------------------------------------------------------------
# Standard WGAN loss
gen_cost = -tf.reduce_mean(disc_fake)
tf.summary.scalar("generator cost",gen_cost)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
tf.summary.scalar("discriminator cost before penalty",disc_cost)
# G_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(G_pro), logits=G_logits))
# D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(G_pro), logits=G_logits))
# D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_pro), logits=D_logits))
# D_loss = D_real_loss + D_fake_loss
# G_loss = G_fake_loss
# loss_sum = tf.summary.scalar("D_loss", D_loss)
# G_loss_sum = tf.summary.scalar("G_loss", G_loss)
# merged_summary_op_d = tf.summary.merge([loss_sum, D_pro_sum])
# merged_summary_op_g = tf.summary.merge([G_loss_sum, G_pro_sum, G_image])

# Gradient penalty on Discriminator 
alpha = tf.random_uniform(shape=[BATCH_SIZE,1], minval=0., maxval=1.)
interpolates = real_data + (alpha*(fake_data - real_data))
gradients = tf.gradients(Discriminator(interpolates, condition_data), [interpolates])[0] #added cond here
for _,grad in gradients: # add a histogram for the gradient
    tf.summary.histogram("{}-grad".format(grad.name.replace(":","_")), grad)
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty
tf.summary.scalar("discriminator cost",disc_cost)

gen_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

# For generating samples: define fixed noise and conditional input
#fixed_real_data, fixed_labels = next(test_generator) 
fixed_real_data, fixed_labels = mnist.test.next_batch(BATCH_SIZE)
fixed_real_data_255 = ((fixed_real_data)*255.).astype('int32') # [0,255] 

if(CONTINUE):
    fixed_noise = tf.get_variable("noise", shape=[BATCH_SIZE, 128]) # get noise from saved model
else:
    fixed_noise = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, 128]), name='noise') #var inst of const: var is saved, dtype float implicit 
tf.summary.tensor_summary("fixed noise for test", fixed_noise)
fixed_noise_samples = Generator(BATCH_SIZE, fixed_labels, noise=fixed_noise)

def generate_image(frame, true_dist):   # generates a batch of samples next to each other in one image!
    samples = session.run(fixed_noise_samples, feed_dict={real_data: fixed_real_data, condition_data: fixed_labels}) # [-1,1]
    #for im in samples: # add a image for the samples
    #    tf.summary.image("{}-image".format(grad.name.replace(":","_")), im)
    samples_255 = ((samples+1.)*(255./2)).astype('int32') # [0,255] 
    samples_01 = ((samples+1.)/2.).astype('float32') # [0,1]
    for i in range(0, BATCH_SIZE):
        samples_255 = np.insert(samples_255, i*2, fixed_real_data_255[i,:], axis=0) # show cond digit next to generated sample
    imsaver.save_images(samples_255.reshape((2*BATCH_SIZE, 1, IM_DIM, IM_DIM)), 'samples_{}.jpg'.format(frame))
    print("Iteration %d : \n" % frame)
    # compare generated to real one
    real = tf.reshape(fixed_real_data, [BATCH_SIZE,IM_DIM,IM_DIM,1])
    pred = tf.reshape(samples_01, [BATCH_SIZE,IM_DIM,IM_DIM,1])
    ssimval = tf.image.ssim(real, pred, max_val=1.0) # tensor batch in, out tensor of ssimvals (64,)
    mseval_per_entry = tf.keras.metrics.mse(real, pred)  # mse on grayscale, on [0,1]
    mseval = tf.reduce_mean(mseval_per_entry, [1,2])
    tf.summary.tensor_summary("SSIM values", ssimval)
    tf.summary.tensor_summary("MSE values", mseval)
    ssimval_list = ssimval.eval()  # to numpy array # (50,)
    mseval_list = mseval.eval() # (50,)
    # print(ssimval_list)
    # print(mseval_list)
    for i in range (0,3):
        plotter.plot('SSIM for sample %d' % (i+1), ssimval_list[i])
        plotter.plot('MSE for sample %d' % (i+1), mseval_list[i])
        print("sample %d \t MSE: %.5f \t SSIM: %.5f \r\n" % (i, mseval_list[i], ssimval_list[i]))
        
        
init_op = tf.global_variables_initializer()  	# op to initialize the variables.
saver = tf.train.Saver()			# ops to save and restore all the variables.

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

merged = tf.summary.merge_all() # delete..
# specify where to save - subdirectories help to filter in tensorboard


# Train loop
with tf.Session(config=config) as session:
    overall_start_time = time.time()
    # Init variables
    if(CONTINUE):
         saver.restore(session, restore_path) # Restore variables from saved session.
         print("Model restored.")
         plotter.restore(START_ITER)  # makes plots start from 0
    else:
         session.run(init_op)	

    summary_writer = tf.summary.FileWriter(logdir, graph = session.graph)

    step = START_ITER # START_ITER: 0 or from last checkpoint
    while(step < ITERS):
    # for iteration in range(START_ITER, ITERS):  
        start_time = time.time()
        
        # if step > 0:
        _data, _labels = mnist.train.next_batch(BATCH_SIZE)
        # batch_z =  np.random.uniform(-1, 1, size=[BATCH_SIZE, Z_DIM]) / # batch_z = np.random.normal(0 , 0.2 , size=[BATCH_SIZE, sample_size])
        _disc_cost, _ , summary_str = sess.run([disc_cost, disc_train_op, merged_summary_op_d],feed_dict={real_data: _data, condition_data = _labels})
        #z: batch_z, y: _labels})
        summary_writer.add_summary(summary_str, step)

        _, summary_str = sess.run([gen_train_op, merged_summary_op_g], feed_dict={condition_data = _labels})
        #z: batch_z, y: _labels})
        summary_writer.add_summary(summary_str, step)

        # _ = session.run(gen_train_op, feed_dict={condition_data: _labels}) # Train generator
        #for i in range(DISC_ITERS):
        # _data, _labels = mnist.train.next_batch(BATCH_SIZE)
        #_disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data: _data, condition_data: _labels}) # Train discriminator
        plotter.plot('train disc cost', _disc_cost)
        iteration_time = time.time() - start_time
        tf.summary.scalar("iteration time",iteration_time) # TODO: need?
        plotter.plot('time', iteration_time)


        if step % 100 == 0:
            D_loss = sess.run(D_loss, feed_dict={real_data: _data, condition_data = _labels}) #z: batch_z, y: _labels})
            fake_loss = sess.run(G_loss, feed_dict={condition_data = _labels}) # z: batch_z, y: _labels})
            print("Step %d: D: loss = %.7f G: loss=%.7f " % (step, D_loss, fake_loss))

        # Calculate dev loss and generate samples every 100 iters
        if step % 100 == 1 and step != 0:
            dev_disc_costs = []
            _data, _labels = mnist.test.next_batch(BATCH_SIZE)
            _dev_disc_cost = session.run([disc_cost], feed_dict={real_data: _data, condition_data: _labels}) #summary = merged
            dev_disc_costs.append(_dev_disc_cost)
            plotter.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(step, _data)
            save_path = saver.save(session, restore_path) # Save the variables to disk.
            print("Model saved in path: %s" % save_path)
            # chkp.print_tensors_in_checkpoint_file("model.ckpt", tensor_name='', all_tensors=True)

            # summaries_writer.add_summary(summary,iteration)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 1):
            plotter.flush()

        plotter.tick()
        step = step + 1

summaries_writer.close() # flushes the outputwriter to disk
        
overall_end_time = time.time()
overall_time = (overall_end_time - overall_start_time)
print("The GAN took ", overall_time, "sec to run")
overall_time /= 60.
print("The GAN took ", overall_time, "min to run")
overall_time /= 60.
print("The GAN took ", overall_time, "h to run")


def test():
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, restore_path)
        sample_z = np.random.uniform(1, -1, size=[BATCH_SIZE, Z_DIM])
        output = sess.run(fake_data, feed_dict={z: sample_z, y: sample_label()})
        save_images(output, [8, 8], './{}/test{:02d}_{:04d}.png'.format(sample_dir, 0, 0))
        # show result of test
        image = cv2.imread('./{}/test{:02d}_{:04d}.png'.format(sample_dir, 0, 0), 0)
        cv2.imshow("test", image)
        cv2.waitKey(-1)
        print("Test finish!")
