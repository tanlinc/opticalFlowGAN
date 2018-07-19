
import os
import os.path
import PIL
import numpy as np
import tflib.UCFdata
import tflib.save_images
#import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from scipy import ndimage
from PIL import Image
from skimage.measure import block_reduce



def down1(image):
    resized_im = block_reduce(image, block_size=(3, 24, 32), func=np.mean)
    np.save(resized_im)


def down2(filename):
    basewidth = 300
    img = Image.open(filename)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save('resized_image2.jpg')


def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res
# e.g., a (100, 200) shape array using a factor of 5 (5x5 blocks) results in a (20, 40) array result:
# ar = np.random.rand(20000).reshape((100, 200))	block_mean(ar, 5).shape # (20,40)
 

def preprocess_image_crop(image_path, img_size):
    '''
    Preprocess the image scaling it so that its smaller size is img_size.
    The larger size is then cropped in order to produce a square image.
    '''
    img = load_img(image_path)
    scale = float(img_size) / min(img.size)
    new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
    # print('old size: %s,new size: %s' %(str(img.size), str(new_size)))
    img = img.resize(new_size, resample=Image.BILINEAR)
    img = img_to_array(img)
    crop_h = img.shape[0] - img_size
    crop_v = img.shape[1] - img_size
    img = img[crop_h:img_size+crop_h, crop_v:img_size+crop_v, :]
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def main():
    #image =  
    #np.save(image)
    inpath = "/home/linkermann/opticalFlow/opticalFlowGAN/data/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g25_c07-0146.jpg"
    outpath = "/home/linkermann/opticalFlow/opticalFlowGAN/data/downsampletest/v_ApplyEyeMakeup_g25_c07-0146"
    image1 = img_to_array(load_img(inpath, grayscale=False, target_size=(320,240), interpolation='nearest'))
    tflib.save_images.save_images(image1, outpath+"-1.jpg")
    image2 = img_to_array(load_img(inpath, grayscale=False, target_size=(32,24), interpolation='nearest'))
    tflib.save_images.save_images(image2, outpath+"-2.jpg")
    image3 = img_to_array(load_img(inpath, grayscale=False, target_size=(40,40), interpolation='nearest'))
    tflib.save_images.save_images(image3, outpath+"-3.jpg")
    image4 = img_to_array(load_img(inpath, grayscale=False, target_size=(40,40), interpolation='lanczos'))
    tflib.save_images.save_images(image4, outpath+"-4.jpg")
    image5 = img_to_array(load_img(inpath, grayscale=False, target_size=(40,40), interpolation='bicubic'))
    tflib.save_images.save_images(image5, outpath+"-5.jpg")

    #down1(image)
    #down2(filename)
    #block_mean(ar, 5)

if __name__ == '__main__':
    main()
