"""
Class for managing our data.
"""
import numpy as np
import random
from glob import glob
from os.path import join, isfile
from tflib.processor import process_image, process_and_crop_image, read_and_crop_flow, read_and_crop_flow_int

class DataSet():

    def __init__(self, image_shape=(32, 32, 3), image_shape_flow=(32, 32, 2)): 
        """Constructor.
        """
        self.data = self.get_data()
        self.image_shape = image_shape
        self.image_shape_flow = image_shape_flow

    @staticmethod
    def get_data():
        """Load our data."""
        # (IS_SERVER):
        root = '/home/linkermann/opticalFlow/opticalFlowGAN/data/SINTEL/training/'
        # save_root = '/home/linkermann/opticalFlow/opticalFlowGAN/data/SINTEL/saved'
        # Desktop
        # root = '/home/linkermann/Desktop/MA/data/SINTEL/training/'
        # save_root = '/home/linkermann/Desktop/MA/data/SINTEL/saved'

        flow_root = join(root, 'flow')
        image_root = join(root, 'clean')  # clean as training?
        validation_image_root = join(root, 'final')	# and final as test?
        flow_paths = join(root, 'flow/sleeping_1/*.flo') # just take 1 category with small flow!
        file_list = glob(flow_paths)
        sorted_file_list= sorted(file_list)
        lenli = len(sorted_file_list) #  should be 49 for sleeping_1

        flow_list = []
        train_image_list = []
        validation_image_list = []

        for i, file in enumerate(sorted_file_list): 
            nextflow = sorted_file_list[(i+1)%lenli]

        #for file in sorted_file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')
            img3 = join(image_root, fprefix + "%04d"%(fnum+2) + '.png')
            validation_img1 = join(validation_image_root, fprefix + "%04d"%(fnum+0) + '.png')
            validation_img2 = join(validation_image_root, fprefix + "%04d"%(fnum+1) + '.png')
            validation_img3 = join(validation_image_root, fprefix + "%04d"%(fnum+2) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(img3) or not isfile(validation_img1) or not isfile(validation_img2) or not isfile(validation_img3) or not isfile(file) or not isfile(nextflow):
                continue

            train_image_list += [[img1, img2, img3]]
            flow_list += [[file, nextflow]]  # add next flow file in list, goes as real flow to discriminator!
            validation_image_list += [[validation_img1, validation_img2, validation_img3]]
            # assert (len(train_image_list) == len(flow_list))

            # turn into numpy
            np_train_image_list = np.array(train_image_list)
            np_flow_list = np.array(flow_list)
            np_validation_image_list = np.array(validation_image_list)
            
        return (np_train_image_list, np_flow_list, np_validation_image_list) # tuple

    def frame_generator(self, batch_size, train_test, concat=False):
        """Return a generator of images that we can use to train on.
        """
        # Get the right dataset for the generator.
        data, flow_data, valid_data = self.data
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
                sequence = [process_and_crop_image(x, self.image_shape) for x in sample] # processor..

                # get corresponding flows
                flow_filenames = flow_data[ri] # shape (2,), 2 flow filenames are in there
                flow_array = [read_and_crop_flow(str(x), self.image_shape_flow) for x in flow_filenames] # returns  np: (2, h*w*2) ?	      

                if concat:
                    # pass sequence back as single array (into an MLP rather than an RNN)
                    sequence = np.concatenate(sequence).ravel()
                    flows = np.concatenate(flow_array).ravel() # need?

                X.append(sequence)  # (n, 6144) -- 3072 + 3072 = two images
                F.append(flows)     # (n, 4096) -- 2048 + 2048 = two flows

            yield (np.array(X), np.array(F))

# (seq_length=1, image_shape=(32, 32, 3), image_shape_flow=(32, 32, 2))
def load_train_gen(batch_size, imageShape, imageShapeFlow):
    data = DataSet(image_shape=imageShape, image_shape_flow=imageShapeFlow)
    return data.frame_generator(batch_size, 'train', concat=True)

def load_test_gen(batch_size, imageShape, imageShapeFlow):
    data = DataSet(image_shape=imageShape, image_shape_flow=imageShapeFlow)
    return data.frame_generator(batch_size, 'validation', concat = True)
