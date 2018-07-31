"""
Process an image that we can pass to our networks.
"""
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
# import tensorflow as tf

def process_image(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, c = target_shape
    image = load_img(image, target_size=(h, w))
    # Turn it into numpy and return.
    img_arr = img_to_array(image)
    #print(img_arr.shape) # (32,32,3)
    #x = (img_arr / 255.).astype(np.float32) # this was the evil line of code..
    x = img_arr.astype(np.uint8)	# cifar returns uint8, is already 0-255 but just to be sure
    x = x.reshape(h,w,c)  # this needs to be added.. do the transpose here!
    x = np.transpose(x, [2,0,1]) #e.g.(3,32,32)
    x = x.reshape(h*w*c,)	# uncomment for 64x64 gan!
    return x
