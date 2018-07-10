import numpy as np

import os
import urllib
import gzip
import pickle
from keras.preprocessing.image import ImageDataGenerator
from tflib.UCFdata import DataSet

data = DataSet()

"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.
Use keras 2+ and tensorflow 1+
Based on:
https://keras.io/preprocessing/image/
"""
def get_generators():
    train_datagen = ImageDataGenerator(	# generates augmented data.
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        './data/train/',
        target_size=(320, 240),
        batch_size=30,
        classes=data.classes,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        './data/test/',
        target_size=(320, 240),
        batch_size=30,
        classes=data.classes,
        class_mode='categorical')

    return train_generator, validation_generator
 #   model.fit_generator(
  #      train_generator,
    #    validation_data=validation_generator,)
    #return model


  #  class_limit = None  # int, can be 1-101 or None
  #  seq_length = 40
  #  load_to_memory = False  # pre-load the sequences into memory
  #  batch_size = 32
  #  nb_epoch = 1000
  #  image_shape = (80, 80, 3)

def generator(train, batch_size=32, seq_length = 40, class_limit=None, image_shape=None,
          load_to_memory=False):
    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        if train:
            X, y = data.get_all_sequences_in_memory(batch_size, 'train', 'images')

            def get_epoch():
                rng_state = np.random.get_state()
                np.random.shuffle(X)
                np.random.set_state(rng_state)
                np.random.shuffle(y)
                for i in xrange(len(images) / batch_size):
                    yield (X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            return get_epoch
        else:
            X_test, y_test = data.get_all_sequences_in_memory(batch_size, 'test', 'images')

            def get_epoch():
                rng_state = np.random.get_state()
                np.random.shuffle(X_test)
                np.random.set_state(rng_state)
                np.random.shuffle(y_test)
                for i in xrange(len(images) / batch_size):
                    yield (X_test[i*batch_size:(i+1)*batch_size], y_test[i*batch_size:(i+1)*batch_size])
            return get_epoch
    else:
        # Get generators.
        if train:
            generator = data.frame_generator(batch_size, 'train', 'images')
        else:
            generator = data.frame_generator(batch_size, 'test', 'images')
        return generator


def load(batch_size):
    return (
        generator(True, batch_size),    # train
        generator(False, batch_size)	# test
    )

def load_train_gen(batch_size):
    return data.frame_generator(batch_size, 'train', 'images')

def load_keras_gen(batch_size):
    train_generator, validation_generator = get_generators()
    return train_generator
