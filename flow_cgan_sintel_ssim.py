import os, sys
sys.path.append(os.getcwd())

import time
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.plot
import tflib.SINTELdataDesktop as sintel
from skimage.measure import compare_ssim as ssim

MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 64 # This overfits substantially; you're probably better off with 64 # or 128?
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 100000 # How many generator iterations to train for # 200000 takes too long
IM_DIM = 32 # number of pixels along x and y (square assumed)
SQUARE_IM_DIM = IM_DIM*IM_DIM # 32*32 = 1024
OUTPUT_DIM = IM_DIM*IM_DIM*3 # Number of pixels (3*32*32) - rgb color
OUTPUT_DIM_FLOW = IM_DIM*IM_DIM*2 # Number of pixels (2*32*32) - uv direction

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def Generator(n_samples, conditions, noise=None):	# input conds additional to noise
    if noise is None:
        noise = tf.random_normal([n_samples, SQUARE_IM_DIM]) 

    noise = tf.reshape(noise, [n_samples, 1, IM_DIM, IM_DIM])
    # new conditional input: last frame
    conds = tf.reshape(conditions, [n_samples, 3, IM_DIM, IM_DIM])  # conditions: (64,3072) TO conds: (64,3,32,32)

    # for now just concat the inputs: noise as fourth dim of cond image 
    output = tf.concat([noise, conds], 1)  # to: (BATCH_SIZE,4,32,32)
    output = tf.reshape(output, [n_samples, SQUARE_IM_DIM*4]) # 32x32x4 = 4096; to: (BATCH_SIZE, 4096)

    output = lib.ops.linear.Linear('Generator.Input', SQUARE_IM_DIM*4, 4*4*4*DIM, output) # 4*4*4*DIM = 64*64 = 4096
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 2, 5, output)  # output flow --> dim is different: 2 instead of 3

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM_FLOW])  # output flow --> dim is different: 2 instead of 3

def Discriminator(inputs, conditions):	# input conds as well
    inputs = tf.reshape(inputs, [-1, 2, IM_DIM, IM_DIM])   # input flow --> dim is different: 2 instead of 3
    conds = tf.reshape(conditions, [-1, 3, IM_DIM, IM_DIM])  # new conditional input: last frame
    # for now just concat the inputs
    ins = tf.concat([inputs, conds], 1) #to: (BATCH_SIZE, 5, 32, 32)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 5, DIM, 5, ins, stride=2)  # first dim is different: 5 instead of 6
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

   #output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM, 8*DIM, 5, output, stride=2)
   # if MODE != 'wgan-gp':
   #     output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,2,3], output)
   # output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*8*DIM]) # adjusted outcome
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*DIM, 1, output)

    return tf.reshape(output, [-1])

cond_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM]) # conditional input for both G and D
cond_data = 2*((tf.cast(cond_data_int, tf.float32)/255.)-.5) #normalized [0,1]!

real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM_FLOW])
real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5) #normalized [0,1]!
fake_data = Generator(BATCH_SIZE, cond_data)

disc_real = Discriminator(real_data, cond_data)
disc_fake = Discriminator(fake_data, cond_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    # Standard WGAN loss
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates, cond_data), [interpolates])[0] #added cond here
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                                                  var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                                                   var_list=lib.params_with_name('Discriminator.'))

# Dataset iterators
gen = sintel.load_train_gen(BATCH_SIZE, (IM_DIM,IM_DIM,3)) # batch size, im size
dev_gen = sintel.load_test_gen(BATCH_SIZE, (IM_DIM,IM_DIM,3))

# For generating samples: define fixed noise and conditional input
fixed_cond_samples, fixed_samples_flow = next(gen)  # shape: (batchsize, 3072) 
fixed_cond_data_int = fixed_cond_samples[:,0:OUTPUT_DIM]  # earlier frame as condition  # shape (64,3072)
#fixed_real_data_int = fixed_cond_samples[:,OUTPUT_DIM:]  # next frame as comparison to result of generator  # shape (64,3072)
fixed_real_data_int = fixed_samples_flow # flow as comparison to result of generator
fixed_cond_data_normalized = 2*((tf.cast(fixed_cond_data_int, tf.float32)/255.)-.5) #normalized [0,1]! 
fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, SQUARE_IM_DIM)).astype('float32'))  # for additional channel
fixed_noise_samples = Generator(BATCH_SIZE, fixed_cond_data_normalized, noise=fixed_noise) # Generator(n_samples,conds, noise):

def mse(x, y):
    return np.linalg.norm(x - y)

def generate_image(frame, true_dist):   # generates 64 (batch-size) samples next to each other in one image!
    samples = session.run(fixed_noise_samples, feed_dict={real_data_int: fixed_real_data_int, cond_data_int: fixed_cond_data_int})
    samples_255 = ((samples+1.)*(255./2)).astype('int32') #back to [0,255] 
    for i in range(0, BATCH_SIZE):
        samples_255= np.insert(samples_255, i*2, fixed_cond_data_int[i],axis=0) # show last frame next to generated sample
    lib.save_images.save_images(samples2show.reshape((2*BATCH_SIZE, 2, IM_DIM, IM_DIM)), 'samples_{}.jpg'.format(frame))
# TODO: does it work with 2 channel images? how to show? save as .flo? extra save_images fct?
    print("Iteration %d : \n" % frame)
    # compare generated to real one
    for i in range(0, BATCH_SIZE):
        real = np.reshape(fixed_real_data_int[i], (IM_DIM,IM_DIM,2))  # use np.reshape! np-array!
        pred = np.reshape(samples[i] , (IM_DIM,IM_DIM,2))  # not samples_255!
        mseval = mse(real, pred)
        ssimval = ssim(real, pred, data_range = pred.max() - pred.min(), multichannel = True) # multichannel instead of grayscale
        print("sample %d \t MSE: %.2f \t SSIM: %.2f \r\n" % (i, mseval, ssimval))
        if (i < 3):
            lib.plot.plot('SSIM for sample %d' % (i+1), ssimval)
            lib.plot.plot('MSE for sample %d' % (i+1), mseval)

# Train loop
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for iteration in range(ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _data, _ = next(gen)  # shape: (batchsize, 6144), double output_dim now   # flow as second argument not needed
            _cond_data = _data[:,0:OUTPUT_DIM] # earlier frame as conditional data, # last frame not needed here
            _ = session.run(gen_train_op, feed_dict={cond_data_int: _cond_data})
        # Train critic
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data, _flow = next(gen)  # shape: (batchsize, 6144), double output_dim now   # flow as second argument
            _cond_data = _data[:,0:OUTPUT_DIM] # earlier frame as conditional data,
            _real_data = _flow           # flow as real data for discriminator

            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_int: _real_data, cond_data_int: _cond_data})
            if MODE == 'wgan':
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            _data, _flow = next(gen)  # shape: (batchsize, 6144), double output_dim now    # flow as second argument
            _cond_data = _data[:,0:OUTPUT_DIM] # earlier frame as conditional data,
            _real_data = _flow           # flow as real data for discriminator
            _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: _real_data, cond_data_int: _cond_data})   
            dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, _data)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()