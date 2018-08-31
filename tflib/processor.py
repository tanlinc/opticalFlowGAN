"""
Process an image that we can pass to our networks.
"""
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from image.io import imread

def randomCrop(img, crop_size):
    th, tw = crop_size
    h = img.shape[0]
    w = img.shape[1]
    h1 = random.randint(0, h - th)
    w1 = random.randint(0, w - tw)
    return img[h1:(h1+th), w1:(w1+tw),:]

def centerCrop(img, crop_size):
    th, tw = crop_size
    h = img.shape[0]
    w = img.shape[1]
    return img[(h-th)//2:(h+th)//2, (w-tw)//2:(w+tw)//2,:]

def process_image(image, target_shape):  # downsampling to desired shape
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, c = target_shape
    image = load_img(image, target_size=(h, w)) # downsampling included!
    # Turn it into numpy and return.
    img_arr = img_to_array(image)
    #print(img_arr.shape) # (32,32,3)
    #x = (img_arr / 255.).astype(np.float32) # this was the evil line of code..
    x = img_arr.astype(np.uint8)	# cifar returns uint8, is already 0-255 but just to be sure
    x = x.reshape(h,w,c)  # this needs to be added.. do the transpose here!
    x = np.transpose(x, [2,0,1]) #e.g.(3,32,32)
    x = x.reshape(h*w*c,)	# uncomment for 64x64 gan!
    return x

def process_and_crop_image(filename, target_shape): # cropping to desired shape
    """Given an image, process it, crop it, and return the array."""
    # Load the image.
    h, w, c = target_shape
    # load image 
    image = imread(filename)
    img_arr = img_to_array(image)  # Turn it into numpy # does it work here??
    img_cropped = centerCrop(img_arr, (h,w))
    #print(img_arr.shape) # check shape (32,32,3)?
    x = img_arr.astype(np.uint8) # return uint8, convert just to be sure
    x = x.reshape(h,w,c)  # transpose here!
    x = np.transpose(x, [2,0,1]) # e.g.(2,32,32)
    x = x.reshape(h*w*c,)	# uncomment for 64x64 gan!
    return x
