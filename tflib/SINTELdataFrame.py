"""
Class for managing our data.
"""
import numpy as np
import random
from glob import glob
from os.path import join, isfile
from tflib.processor import process_image, process_and_crop_image #, randomCrop, centerCrop

class DataSet():

    def __init__(self, image_shape=(32, 32, 3)): 
        """Constructor.
        """
        self.data = self.get_data()
        self.image_shape = image_shape

    @staticmethod
    def get_data():
        """Load our data."""
        # (IS_SERVER):
        root = '/home/linkermann/opticalFlow/opticalFlowGAN/data/SINTEL/training/'
        save_root = '/home/linkermann/opticalFlow/opticalFlowGAN/data/SINTEL/saved'
        # Desktop
        # root = '/home/linkermann/Desktop/MA/data/SINTEL/training/'
        # save_root = '/home/linkermann/Desktop/MA/data/SINTEL/saved'

        if isfile(join(save_root, 'train.npy')) and isfile(join(save_root, 'flow.npy')) and isfile(join(save_root, 'validation.npy')):
            print("load lists from files")
            np_train_image_list = np.load(join(save_root, 'train.npy'))
            np_flow_list = np.load(join(save_root, 'flow.npy'))
            np_validation_image_list = np.load(join(save_root, 'validation.npy'))
            return (np_train_image_list, np_flow_list, np_validation_image_list) # tuple

        flow_root = join(root, 'flow')
        image_root = join(root, 'clean')
        validation_image_root = join(root, 'final')
        flow_paths = join(root, 'flow/*/*.flo')
        file_list = glob(flow_paths)
        sorted_file_list= sorted(file_list)
        lenli = len(sorted_file_list) # 1041

        flow_list = []
        train_image_list = []
        validation_image_list = []

        for i, file in enumerate(sorted_file_list):
            nextflow = sorted_file_list[(i+1)%lenli]
            cat1 = get_category_from_path(file)
            cat2 = get_category_from_path(nextflow)
            if not cat1 == cat2:
                print("discarding because category of first frame is " + cat1 + " and next is " + cat2)
                continue

        #for file in sorted_file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')
            validation_img1 = join(validation_image_root, fprefix + "%04d"%(fnum+0) + '.png')
            validation_img2 = join(validation_image_root, fprefix + "%04d"%(fnum+1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(validation_img1) or not isfile(validation_img2) or not isfile(file) or not isfile(nextflow):
                continue

            train_image_list += [[img1, img2]]
            flow_list += [[file, nextflow]]  # add next flow file in list (if same category), goes as real flow to discriminator!
            validation_image_list += [[validation_img1, validation_img2]]
            assert (len(train_image_list) == len(flow_list))

            # turn into numpy
            np_train_image_list = np.array(train_image_list)
            np_flow_list = np.array(flow_list)
            np_validation_image_list = np.array(validation_image_list)

            # save as npy file to load from now on
            np.save(join(save_root, 'train.npy'), np_train_image_list)
            np.save(join(save_root, 'flow.npy'), np_flow_list)
            np.save(join(save_root, 'validation.npy'), np_validation_image_list)
            
        return (np_train_image_list, np_flow_list, np_validation_image_list) # tuple

    def frame_generator(self, batch_size, train_test, concat=False):
        """Return a generator of images that we can use to train on.
        """
        # Get the right dataset for the generator.
        data, _ , valid_data = self.data
        if(train_test == 'validation'):
            data = valid_data
        # print(data.shape) # (48, 2)
        # print(data[0].shape) # (2,)

        print("Creating %s generator with %d samples." % (train_test, len(data)))  # should be 1018
      
        while True:
            X = []
            F = []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Get a random sample.  ## TODO: is it a problem to always take random samples?
                ri = random.randint(0, len(data) - 1) # random integer s.t. a <= N <= b.
                sample = data[ri] # shape (2,), 2 frame filenames are in there
  
                # Given a set of filenames, build a sequence.
                sequence = [process_image(x, self.image_shape) for x in sample] # change to crop?      

                if concat:
                    # pass sequence back as single array (into an MLP rather than an RNN)
                    sequence = np.concatenate(sequence).ravel()

                X.append(sequence)  # (n, 6144) -- 3072 + 3072 = two images

            yield (np.array(X), np.array(F))

def get_category_from_path(path):
    parts = path.split('/')
    return parts[-2]

# (seq_length=1, image_shape=(32, 32, 3))
def load_train_gen(batch_size, imageShape):
    data = DataSet(image_shape=imageShape)
    return data.frame_generator(batch_size, 'train', concat=True)

def load_test_gen(batch_size, imageShape):
    data = DataSet(image_shape=imageShape)
    return data.frame_generator(batch_size, 'validation', concat = True)
