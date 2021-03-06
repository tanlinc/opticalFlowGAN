"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import pandas as pd
import sys
import operator
from tflib.processor import process_image
from keras.utils import np_utils

class DataSet():

    def __init__(self, seq_length=1, class_limit=1, image_shape=(32, 32, 3)): 
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = '/home/linkermann/Desktop/MA/data/sequences/'
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data():
        """Load our data from file."""
        with open('/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/data/data_file_selected.csv', 'r') as fin:  # changed file to selected
            reader = csv.reader(fin)
            data = list(reader)

        return data

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = np_utils.to_categorical(label_encoded, len(self.classes))
        label_hot = label_hot[0]  # just get a single row

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, batch_Size, train_test, data_type, concat=False):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Getting %s data with %d samples." % (train_test, len(data)))

        X, y = [], []
        for row in data:

            sequence = self.get_extracted_sequence(data_type, row)

            if sequence is None:
                print("Can't find sequence. Did you generate them?")
                raise

            if concat:
                # We want to pass the sequence back as a single array. This
                # is used to pass into a CNN or MLP, rather than an RNN.
                sequence = np.concatenate(sequence).ravel()

            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    def frame_generator(self, batch_size, train_test, data_type, concat=False):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))
      
        while True:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)
                # print(sample)

                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length, 3)  # number determines skip behaviour

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)
                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    sys.exit()  # TODO this should raise

                if concat:
                    # We want to pass the sequence back as a single array. This
                    # is used to pass into an MLP rather than an RNN.
                    sequence = np.concatenate(sequence).ravel()

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield (np.array(X), np.array(y))

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]		

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = self.sequence_path + filename + '-' + str(self.seq_length) + \
            '-' + data_type + '.txt'
        if os.path.isfile(path):
            # Use a dataframe/read_csv for speed increase over numpy.
            features = pd.read_csv(path, sep=" ", header=None)
            return features.values
        else:
            return None

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = '/home/linkermann/Desktop/MA/data/' + sample[0] + '/' + sample[1] + '/'
        filename = sample[2]
        # print(filename) # to see which class is taken
        images = sorted(glob.glob(path + filename + '*.jpg'))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split('/')
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size, skipNum):		
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the original list."""
        assert len(input_list) >= size
      
        ### this skips more frames in between if larger number of frames for clip   	 
        if(skipNum == 1):
            # Get the number to skip between iterations.
            skip = len(input_list) // size
            # Build our new output.
            output = [input_list[i] for i in range(0, len(input_list), skip)]

        ### this skips beginning or end of sequence, no frames in between
        if(skipNum == 2):
            rdm = random.randint(0, len(input_list) - size)
            output = input_list[rdm:rdm+size]
        
        else: # skip every 2nd frame ..then random start .. 
            filtered_list = [input_list[i] for i in range(0, len(input_list), 2)]
            rdm = random.randint(0, len(filtered_list) - size) #should never be problem that too few are left
            output = filtered_list[rdm:rdm+size]
            # output = filtered_list[0:size] # always start from beginning

        # Cut off the last one if needed.
        return output[:size]


# change size of input image here..
# change seq_length here to get more than 1 frame as input
# (seq_length=1, class_limit=1, image_shape=(32, 32, 3))
def load_train_gen(batch_size, seqLength, classLimit, imageShape):
    data = DataSet(seq_length=seqLength, class_limit=classLimit, image_shape=imageShape)
    return data.frame_generator(batch_size, 'train', 'images', concat=True)

def load_test_gen(batch_size, seqLength, classLimit, imageShape):
    data = DataSet(seq_length=seqLength, class_limit=classLimit, image_shape=imageShape)
    return data.frame_generator(batch_size, 'test', 'images', concat = True)
