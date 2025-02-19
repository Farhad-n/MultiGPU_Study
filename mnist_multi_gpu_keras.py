'''Trains a simple convnet on the MNIST dataset.
Gets to 99.35% test accuracy after 10 epochs
12 seconds per epoch on a Gforce 1080 TI GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1960)  # for reproducibility

from tensorflow.contrib.keras.api.keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
tensorboard = TensorBoard(log_dir='/home/norman/MNIST_train', histogram_freq=1,
                          write_graph=True, write_images=False, embeddings_freq=1)
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--extras', help='(absolute) path to keras-extras')
parser.add_argument('--gpus', help='number of GPUs')
parser.print_help()
args = parser.parse_args()

import sys
sys.path.append(args.extras)

from multi_gpu import make_parallel

ngpus = int(args.gpus)
#ngpus = int(1)
print("Using %i GPUs" %ngpus)

batch_size = 128
nb_classes = 10
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(256, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(128, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

if ngpus > 1:
    model = make_parallel(model,ngpus)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

start_time = time.time()
model.fit(X_train, Y_train, batch_size=batch_size*ngpus, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))#, callbacks=[tensorboard])
score = model.evaluate(X_test, Y_test, verbose=0)
model.summary()
print('Test score:', score[0])
print('Test accuracy:', score[1])
duration = time.time() - start_time
print('Total Duration (%.3f sec)' % duration)
