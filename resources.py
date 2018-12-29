#!/usr/bin/env python

from keras.applications import VGG16
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import data_resources
from keras.preprocessing.image import load_img

n_epochs = 50  # 50
validation_split = 0.3

batch_size = 15   # 32       #treinar com 20
image_size = 100  # 128     #treinar com 350


def prepare_datasets(train_directory):

    min = 1000000
    max = 0

    for n_inst in data_resources.n_instances_info:
        if(n_inst > max):
            max = n_inst
        elif(n_inst < min):
            min = n_inst

    n_validation = round(min * validation_split, 0)
    n_training = min - n_validation

    # iterate classes informations .txt
    for info_path in data_resources.data_info:

        with open(info_path) as f:
            content = f.readlines()
            content = [x.strip() for x in content]

            # iterate each line
            for instance_info in content:

                parts = instance_info.split(" ")
                image_path = parts[0]
                y = parts[1]

                print(image_path)
                print(y)

    '''
    # Step 1.3: Give random transformations to sets
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    #test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_directory,
        subset='training',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = train_datagen.flow_from_directory(
        train_directory,
        subset='validation',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

    

    return train_generator, test_generator
    '''
