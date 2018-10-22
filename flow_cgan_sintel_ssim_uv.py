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
import tflib.flow_handler as fh
import tflib.SINTELdata as sintel

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
CONTINUE = False  # Default False, set True if restoring from checkpoint
START_ITER = 0  # Default 0, set accordingly if restoring from checkpoint (100, 200, ...)
CURRENT_PATH = "sintel/flowcganuv"

restore_path = "/home/linkermann/opticalFlow/opticalFlowGAN/results/" + CURRENT_PATH + "/model.ckpt"

lib.print_model_settings(locals().copy())

if(CONTINUE):
    tf.reset_default_graph()

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
    # new conditional input: last frames
    conds = tf.reshape(conditions, [n_samples, 6, IM_DIM, IM_DIM])  # conditions: (64,2*3072) TO conds: (64,6,32,32)

    # for now just concat the inputs: noise as seventh dim of cond image 
    output = tf.concat([noise, conds], 1)  # to: (BATCH_SIZE,7,32,32)
    output = tf.reshape(output, [n_samples, SQUARE_IM_DIM*7]) # 32x32x4 = 4096; to: (BATCH_SIZE, 4096)

    output = lib.ops.linear.Linear('Generator.Input', SQUARE_IM_DIM*7, 4*4*4*DIM, output) # 4*4*4*DIM = 64*64 = 4096
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 2, 5, output)  # output flow in color --> dim is 2

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM_FLOW])  # output flow --> dim is 2

def Discriminator(inputs, conditions):	# input conds as well
    inputs = tf.reshape(inputs, [-1, 2, IM_DIM, IM_DIM])   # input flow --> dim is 2
    conds = tf.reshape(conditions, [-1, 6, IM_DIM, IM_DIM])  # new conditional input: last frames
    # for now just concat the inputs
    ins = tf.concat([inputs, conds], 1) #to: (BATCH_SIZE, 8, 32, 32)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 8, DIM, 5, ins, stride=2)  # first dim is different: 8 now
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

cond_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 2*OUTPUT_DIM]) # cond input for G and D, 2 frames!
cond_data = 2*((tf.cast(cond_data_int, tf.float32)/255.)-.5) #normalized [-1,1]!

#real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM_FLOW]) # real data is flow, dim 2!
real_data =  tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM_FLOW]) #already float, normalized [-1,1]!
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
gen = sintel.load_train_gen(BATCH_SIZE, (IM_DIM,IM_DIM,3), (IM_DIM,IM_DIM,2)) # batch size, im size, im size flow
dev_gen = sintel.load_test_gen(BATCH_SIZE, (IM_DIM,IM_DIM,3), (IM_DIM,IM_DIM,2))

# For generating samples: define fixed noise and conditional input
fixed_cond_samples, fixed_flow_samples = next(gen)  # shape: (batchsize, 3072) 
fixed_cond_data_int = fixed_cond_samples[:,0:2*OUTPUT_DIM] # earlier frames as condition, cond samples shape (64,3*3072)
fixed_real_data = fixed_flow_samples[:,OUTPUT_DIM_FLOW:]	 # later flow for discr, flow samples shape (64,2048)
fixed_real_data_norm01 = tf.cast(fixed_real_data+1.0, tf.float32)/2.0 # [0,1]
fixed_cond_data_normalized = 2*((tf.cast(fixed_cond_data_int, tf.float32)/255.)-.5) #normalized [-1,1]! 
if(CONTINUE):
    fixed_noise = tf.get_variable("noise", shape=[BATCH_SIZE, SQUARE_IM_DIM]) # take same noise like saved model
else:
    fixed_noise = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, SQUARE_IM_DIM], dtype=tf.float32), name='noise') #variable: saved, for additional channel
fixed_noise_samples = Generator(BATCH_SIZE, fixed_cond_data_normalized, noise=fixed_noise) # Generator(n_samples,conds, noise):

def generate_image(frame, true_dist):   # generates 64 (batch-size) samples next to each other in one image!
    print("Iteration %d : \n" % frame)
    samples = session.run(fixed_noise_samples, feed_dict={real_data: fixed_real_data, cond_data_int: fixed_cond_data_int}) # output range (-1.0,1.0), size=(BATCH_SIZE, OUT_DIM)
    #samples_255 = ((samples+1.)*(255./2)).astype('int32') #(-1,1) to [0,255] for displaying
    samples_01 = ((samples+1.)/2.).astype('float32') # [0,1]
    samples_255 = np.zeros((2*BATCH_SIZE, OUTPUT_DIM))
    sample_flowimages, real_flowimages = [], []
    for i in range(0, BATCH_SIZE):
        real_flowimg, flowimg = [],[] # reset to be sure
        flowimg = fh.computeFlowImg(samples[i].reshape((IM_DIM,IM_DIM,2)))    # (32, 32, 3) # now color img!! :)
        flowimage_T = np.transpose(flowimg, [2,0,1])  #  (3, 32, 32)
        flowimage = flowimage_T.reshape((OUTPUT_DIM,))  # instead of flatten?
        sample_flowimages.append(flowimage)
        real_flowimg = fh.computeFlowImg(fixed_real_data[i].reshape((IM_DIM,IM_DIM,2))) 
        real_flowimage_T = np.transpose(real_flowimg, [2,0,1])  #  (3, 32, 32)
        real_flowimage = real_flowimage_T.reshape((OUTPUT_DIM,))  # instead of flatten? 
        real_flowimages.append(real_flowimage)
        samples_255[2*i+1,:] = flowimage.astype('int32') # sample flow color image   
        last_frame = fixed_cond_data_int[i,OUTPUT_DIM:].astype('int32')   # need to transpose??
        samples_255[2*i,:] = last_frame # last frame left of generated sample
        # samples_255= np.insert(samples_255, i*2, fixed_cond_data_int[i],axis=0)

    lib.save_images.save_images(samples_255.reshape((2*BATCH_SIZE, 3, IM_DIM, IM_DIM)), 'samples_{}.jpg'.format(frame)) # also save as .flo?
    sample_flowims_np = np.asarray(sample_flowimages, np.int32)
    real_flowims_np = np.asarray(real_flowimages, np.int32)
    sample_flowims = tf.convert_to_tensor(sample_flowims_np, np.int32)
    real_flowims = tf.convert_to_tensor(real_flowims_np, np.int32) # turn into tensor to reshape later
    # tensor = tf.constant(np_array)

    # compare generated flow to real one 	# float..?
    # u-v-component wise
    real = tf.reshape(fixed_real_data_norm01, [BATCH_SIZE,IM_DIM,IM_DIM,2])  # use tf.reshape! Tensor! batch!
    real_u, real_v = real[:,:,:,0], real[:,:,:,1] # use some tf funct here?
    pred = tf.reshape(samples_01,[BATCH_SIZE,IM_DIM,IM_DIM,2])  # use tf reshape! and not samples2show!
    pred_u, pred_v = pred[:,:,:,0], pred[:,:,:,1] # shape (64, 32, 32)
    print((real_u.eval()).shape)
    print(real_u.eval())
    print((real_v.eval()).shape) 
    print(real_v.eval())
    print((pred_u.eval()).shape)
    print(pred_u.eval())
    print((pred_v.eval()).shape)
    print(pred_v.eval())
    # print((real_u.flatten()).shape)
    real_u_flat = tf.reshape(real_u, [64, -1])
    pred_u_flat = tf.reshape(pred_u, [64, -1])

    # mse & ssim on components
    mseval_per_entry_u = tf.keras.metrics.mse(real_u, pred_u)  #  on grayscale, on [0,1].. # shape (64,32) # flatten??
    print((mseval_per_entry_u.eval()).shape)
    mseval_per_entry_u_flat = tf.keras.metrics.mse(real_u_flat, pred_u_flat)  #  on grayscale, on [0,1].. # shape (64,32) # flatten??
    print((mseval_per_entry_u_flat.eval()).shape)
    mseval_u_flat = tf.reduce_mean(mseval_per_entry_u_flat, 1)
    print((mseval_u_flat.eval()).shape)
    print(mseval_u_flat.eval())
    mseval_u = tf.reduce_mean(mseval_per_entry_u, 1) 
    mseval_per_entry_v = tf.keras.metrics.mse(real_v, pred_v)  #  on grayscale, on [0,1]..
    mseval_v = tf.reduce_mean(mseval_per_entry_v, 1)
    ssimval_u = tf.image.ssim(real_u, pred_u, max_val=1.0)  # in: tensor 64-batch, out: tensor ssimvals (64,)
    ssimval_v = tf.image.ssim(real_v, pred_v, max_val=1.0)  # in: tensor 64-batch, out: tensor ssimvals (64,)
    # avg: add and divide by 2    
    print(ssimval_u.eval())  # 0.07398668 # 0.1966
    print(ssimval_v.eval())  # 0.059839506 # 0.2009
    mseval_uv = tf.add(mseval_u, mseval_v)  # tf.cast neccessary?
    tensor2 = tf.constant(2.0, shape=[64, 1])
    ssimval_uv = tf.add(ssimval_u, ssimval_v)
    print(ssimval_uv.eval()) # 0.13382618 # 0.3976
    mseval_uv = tf.div(mseval_uv, tensor2)
    ssimval_uv = tf.div(ssimval_uv, tensor2)
    print(ssimval_uv.eval()) # 0.1988 times 64 ??
    ssimval_list_uv = ssimval_uv.eval()  # to numpy array # (64,)
    mseval_list_uv = mseval_uv.eval() # (64,)

    # flow color ims to gray
    real_flowims = tf.cast(real_flowims, tf.float32)/255. # to [0,1]
    real_color = tf.reshape(real_flowims, [BATCH_SIZE,IM_DIM,IM_DIM,3]) 
    real_gray = tf.image.rgb_to_grayscale(real_color) # tensor batch to gray; returns original dtype = float [0,1]
    print((real_gray.eval()).shape)
    sample_flowims = tf.cast(sample_flowims, tf.float32)/255. # to [0,1]
    pred_color = tf.reshape(sample_flowims, [BATCH_SIZE,IM_DIM,IM_DIM,3])  # use tf.reshape! Tensor! batch!
    pred_gray = tf.image.rgb_to_grayscale(pred_color)

    # mse & ssim on grayscale
    mseval_per_entry_rgb = tf.keras.metrics.mse(real_gray, pred_gray)  #  on grayscale, on [0,1]..
    mseval_rgb = tf.reduce_mean(mseval_per_entry_rgb, [1,2])
    ssimval_rgb = tf.image.ssim(real_gray, pred_gray, max_val=1.0)  # in: tensor 64-batch, out: tensor ssimvals (64,)
    ssimval_list_rgb = ssimval_rgb.eval()  # to numpy array # (64,)
    mseval_list_rgb = mseval_rgb.eval() # (64,)

    # print(ssimval_list)
    # print(mseval_list)
    for i in range (0,3):
        lib.plot.plot('SSIM uv for sample %d' % (i+1), ssimval_list_uv[i])
        lib.plot.plot('SSIM rgb for sample %d' % (i+1), ssimval_list_rgb[i])
        lib.plot.plot('MSE uv for sample %d' % (i+1), mseval_list_uv[i])
        lib.plot.plot('MSE rgb for sample %d' % (i+1), mseval_list_rgb[i])
        print("sample %d \t MSE: %.5f \t %.5f \t SSIM: %.5f \t %.5f\r\n" % (i, mseval_list_uv[i], mseval_list_rgb[i], ssimval_list_uv[i], ssimval_list_rgb[i]))
    

init_op = tf.global_variables_initializer()  	# op to initialize the variables.
saver = tf.train.Saver()			# ops to save and restore all the variables.

# Train loop
with tf.Session() as session:
    if(CONTINUE):
         # Restore variables from disk.
         saver.restore(session, restore_path)
         print("Model restored.")
         lib.plot.restore(START_ITER)  # does not fully work, but makes plots start from newly started iteration
    else:
         session.run(init_op)		

    for iteration in range(START_ITER, ITERS):  # START_ITER: 0 or from last checkpoint
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _data, _ = next(gen)  # shape: (batchsize, 6144), double output_dim now   # flow as second argument not needed
            _cond_data = _data[:, 0:2*OUTPUT_DIM] # earlier frames as conditional data, # flow for disc not needed here
            _ = session.run(gen_train_op, feed_dict={cond_data_int: _cond_data})
        # Train critic
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data, _flow = next(gen)  # shape: (batchsize, 6144), double output_dim now   # flow as second argument
            _cond_data = _data[:, 0:2*OUTPUT_DIM]	# earlier 2 frames as conditional data,
            _real_data = _flow[:,OUTPUT_DIM_FLOW:] 	# later flow as real data for discriminator

            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data: _real_data, cond_data_int: _cond_data})
            if MODE == 'wgan':
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            _data, _flow = next(gen)  # shape: (batchsize, 6144), double output_dim now    # flow as second argument
            _cond_data = _data[:, 0:2*OUTPUT_DIM] # earlier 2 frames as conditional data
            _real_data = _flow[:,OUTPUT_DIM_FLOW:] 	# later flow as real data for discriminator
            _dev_disc_cost = session.run(disc_cost, feed_dict={real_data: _real_data, cond_data_int: _cond_data})   
            dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, _data)

            # Save the variables to disk.
            save_path = saver.save(session, restore_path)
            print("Model saved in path: %s" % save_path)
            # chkp.print_tensors_in_checkpoint_file("model.ckpt", tensor_name='', all_tensors=True)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
