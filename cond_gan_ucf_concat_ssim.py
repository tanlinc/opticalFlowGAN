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
import tflib.UCFdataEasy as UCFdata
from skimage import img_as_float, img_as_ubyte
from skimage.measure import compare_ssim as ssim
from skimage.color import rgb2gray

MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 64 # This overfits substantially; you're probably better off with 64 # or 128?
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 100000 # How many generator iterations to train for # 200000 takes too long
OUTPUT_DIM = 3072 # Number of pixels in UCF101 (3*32*32)

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
        noise = tf.random_normal([n_samples, 1024]) # 32*32 = 1024

    noise = tf.reshape(noise, [n_samples, 1, 32, 32])
    # new conditional input: last frame
    conds = tf.reshape(conditions, [n_samples, 3, 32, 32])  # conditions: (64,3072) TO conds: (64,3,32,32)

    # for now just concat the inputs: noise as fourth dim of cond image 
    output = tf.concat([noise, conds], 1)  # to: (BATCH_SIZE,4,32,32)
    output = tf.reshape(output, [n_samples, 4096]) # 32x32x4 = 4096; to: (BATCH_SIZE, 4096)

    output = lib.ops.linear.Linear('Generator.Input', 4096, 4*4*4*DIM, output) # 4*4*4*DIM = 64*64 = 4096
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, conditions):	# input conds as well
    inputs = tf.reshape(inputs, [-1, 3, 32, 32])
    conds = tf.reshape(conditions, [-1, 3, 32, 32])  # new conditional input: last frame
    # for now just concat the inputs
    ins = tf.concat([inputs, conds], 1) #to: (BATCH_SIZE, 6, 32, 32)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 6, DIM, 5, ins, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)  # (5,5,64,128) resource exhausted error
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

real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
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
gen = UCFdata.load_train_gen(BATCH_SIZE, 2, 2, (32,32,3)) # batch size, seq len, #classes, im size
dev_gen = UCFdata.load_test_gen(BATCH_SIZE, 2, 2, (32,32,3))

# For generating samples: define fixed noise and conditional input
fixed_cond_samples, _ = next(gen)  # shape: (batchsize, 3072)
fixed_cond_data_int = fixed_cond_samples[:,0:3072]  # earlier frame as condition  # shape (64,3072)
fixed_real_data_int = fixed_cond_samples[:,3072:]  # next frame as comparison to result of generator  # shape (64,3072)
fixed_cond_data_normalized = 2*((tf.cast(fixed_cond_data_int, tf.float32)/255.)-.5) #normalized [0,1]! 
fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 1024)).astype('float32'))  # for additional channel: 32*32 = 1024
fixed_noise_samples = Generator(BATCH_SIZE, fixed_cond_data_normalized, noise=fixed_noise) # Generator(n_samples,conds, noise):
file = open(“ssimfile.txt”,”w+”)  # a file for storing the mse and ssim values

def mse(x, y):
    return np.linalg.norm(x - y)

def generate_image(frame, true_dist):   # generates 64 (batch-size) samples next to each other in one image!
    samples = session.run(fixed_noise_samples, feed_dict={cond_data_int: fixed_cond_data_int})
    samples = ((samples+1.)*(255./2)).astype('int32') #back to [0,255] 
    print(samples.shape)
    lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 32, 32)), 'samples_{}.jpg'.format(frame))
    print(samples.shape)
    file.write("Iteration %d :" % frame)
    # compare generated to real one
    for i in range(0,64):
        real = tf.reshape(fixed_real_data_int[i,:], [32,32,3]) 
        x = img_as_ubyte(rgb2gray(img_as_float(real))) # fixed_real_data_int
        pred = tf.reshape(samples[i,:], [32,32,3]) 
        y = img_as_ubyte(rgb2gray(img_as_float(pred)))  # samples
        # to 0-255 for mse calculation
        mse = mse(y, x)
        ssim = ssim(y, x, data_range=x.max() - x.min())
        file.write("sample %d \t MSE: %.2f \t SSIM: %.2f \r\n" % i, mse, ssim) 

# Train loop
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for iteration in range(ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)
        # Train critic
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data, _ = next(gen)  # shape: (batchsize, 6144) ##not 3072 anymore
            # extract real and cond data
            _cond_data = _data[:,0:3072] # earlier frame as conditional data,
            _real_data = _data[:,3072:] # last frame as real data for discriminator

            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_int: _real_data, cond_data_int: _cond_data})
            if MODE == 'wgan':
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            images, _ = next(dev_gen)
            _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: images[1,:,:], cond_data_int: images[0,:,:]})    			# earlier frame as condition
            dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, _data)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()

file.close() 
