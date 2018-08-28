"""
Class for managing our data.
"""
import numpy as np
import random
from glob import glob
from os.path import join, isfile
from tflib.processor import process_image

class DataSet():

    def __init__(self, image_shape=(32, 32, 3)): 
        """Constructor.
        """
        self.data = self.get_data()
        self.image_shape = image_shape

    @staticmethod
    def get_data():
        """Load our data."""
        root = '/home/linkermann/Desktop/MA/data/SINTEL/training/'
        flow_root = join(root, 'flow')
        image_root = join(root, 'clean')
        validation_image_root = join(root, 'final')
        flow_paths = join(root, 'flow/*/*.flo')
        file_list = glob(flow_paths)
        sorted_file_list= sorted(file_list)

        flow_list = []
        train_image_list = []
        validation_image_list = []

        for file in sorted_file_list:
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

            if not isfile(img1) or not isfile(img2) or not isfile(validation_img1) or not isfile(validation_img2) or not isfile(file):
                continue

            train_image_list += [[img1, img2]]
            flow_list += [file]
            validation_image_list += [[validation_img1, validation_img2]]

        size = len(train_image_list)
        # frame_size = train_image_list[0][0].shape

        assert (len(train_image_list) == len(flow_list))
        return [train_image_list, flow_list, validation_image_list] # or rather tuple?

    def frame_generator(self, batch_size, train_test, concat=False):
        """Return a generator of images that we can use to train on.
        """
        # Get the right dataset for the generator.
        data = self.data[0] if train_test == 'train' else self.data[2]
        flow_data = self.data[1]

        print("Creating %s generator with %d samples." % (train_test, len(data)))
      
        while True:
            X = []
            F = []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # TODO Get a random sample.
                ri = random.randint(0, len(data)) # random integer s.t. a <= N <= b.
                sample = data[ri]
                flow = flow_data[ri]
                sequence = self.build_image_sequence(sample)
                
                if concat:
                    # We want to pass the sequence back as a single array. This
                    # is used to pass into an MLP rather than an RNN.
                    sequence = np.concatenate(sequence).ravel()

                X.append(sequence)
                F.append(flow)

            yield (np.array(X), np.array(F))

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]		

# (seq_length=1, image_shape=(32, 32, 3))
def load_train_gen(batch_size, imageShape):
    data = DataSet(image_shape=imageShape)
    return data.frame_generator(batch_size, 'train', concat=True)

def load_test_gen(batch_size, imageShape):
    data = DataSet(image_shape=imageShape)
    return data.frame_generator(batch_size, 'validation', concat = True)
