from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tflib.save_images

def process_image(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x


inpath = "/home/linkermann/Desktop/MA/data/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01-0008.jpg"
outpath = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/data/downsampletest/v_ApplyEyeMakeup_g08_c01-0008"

image1 = process_image(inpath, (320,240,3))
image1  = np.transpose(image1, [2,0,1])
tflib.save_images.save_images(image1.reshape((1,3,320,240)), outpath+"-1.jpg")

image2 = process_image(inpath, (32,32,3))
image2  = np.transpose(image2, [2,0,1])
tflib.save_images.save_images(image2.reshape((1,3,32,32)), outpath+"-2.jpg")

image3 = process_image(inpath, (40,40,3))
image3  = np.transpose(image3, [2,0,1])
tflib.save_images.save_images(image3.reshape((1,3,40,40)), outpath+"-3.jpg")

image4 = process_image(inpath, (64,64,3))
image4  = np.transpose(image4, [2,0,1])
tflib.save_images.save_images(image4.reshape((1,3,64,64)), outpath+"-4.jpg")
