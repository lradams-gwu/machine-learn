# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 07:36:03 2017

@author: Leon Adams
"""
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
import numpy as np
NUM_TO_AUGMENT=5

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# Constants
BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# One hot encoding and normalization

y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

# augmentation
print("Augmenting training set images")
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')

xtas, ytas = [], []
for i in range(X_train.shape[0]):
    num_aug = 0
    x = X_train[i]
    x = x[np.newaxis,:,:,:]
    for x_aug in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cifar', save_format='jpeg'):
        if num_aug >= NUM_TO_AUGMENT:
            break
        xtas.append(x_aug[0])

        num_aug += 1


# Set up network
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3, 3), padding="same", input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()

# fit the dataget
datagen.fit(X_train)
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
# train
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), samples_per_epoch=X_train.shape[0],
                              epochs=NB_EPOCH, verbose=VERBOSE)