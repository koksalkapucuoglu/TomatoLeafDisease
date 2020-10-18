#######################################################################################

import os 
import sys
import random
import json
import zipfile
from zipfile import ZipFile

print("[INFO]: Imported primary library")

#######################################################################################
print("[INFO]: Extracting dataset from zip file ...")
file_name = '/content/drive/kaggle/tomato_dataset1/tomato.zip'

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print("[INFO]: Extracted dataset zip file")
  
#######################################################################################

import tensorflow as tf
#tf.test.gpu_device_name()
state_gpu = tf.test.gpu_device_name()
print("[INFO]: GPU usage{0}".format(state_gpu))

#######################################################################################

#import tensorflow and kaggle library
import keras_preprocessing 
from tensorflow.keras.preprocessing import image
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

print("[INFO]: Imported keras and tensorflow library")
print("[INFO]: Tensorflow version{}".format(tf.__version__))

#######################################################################################

# ImageDataGenerator and get train data and validation data
TRAINING_DIR = '/content/New Plant Diseases Dataset(Augmented)/train/' 
VALIDATION_DIR = '/content/New Plant Diseases Dataset(Augmented)/valid/'

print("[INFO]: Training DIR{}".format(TRAINING_DIR))
print("[INFO]: Validation DIR{}".format(VALIDATION_DIR))

#######################################################################################

print("[INFO]: Initialize ImageDataGenerator")

# this is the augmentation configuration we will use for training
train_gen = ImageDataGenerator(
rescale = 1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

valid_gen = ImageDataGenerator(rescale = 1./255)
 
#######################################################################################

print("[INFO]: Training Batch Size:",128)
print("[INFO]: Validation Batch Size:",32)

train_data = train_gen.flow_from_directory(
TRAINING_DIR,
target_size=(227,227),
class_mode='categorical',
color_mode="rgb",
batch_size=128
)

valid_data = valid_gen.flow_from_directory(
VALIDATION_DIR,
target_size=(227,227),
class_mode='categorical',
color_mode="rgb"
)


#######################################################################################

print("[INFO]: Classes of Dataset")

for cl_indis, cl_name in enumerate(train_data.class_indices):
     print(cl_indis, cl_name)

#######################################################################################

# import keras libary for create cnn layer 
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.layers.normalization import BatchNormalization

print("[INFO]: Imported CNN library")

#######################################################################################

# Initializing the CNN based AlexNet
model = Sequential()

#valid:zero padding, same:keep same dimensionality by add padding

# Convolution Step 1
model.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(227, 227, 3), activation = 'relu'))

# Max Pooling Step 1
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid'))

# Convolution Step 2
model.add(Convolution2D(256, 5, strides = (1, 1), padding='same', activation = 'relu'))

# Max Pooling Step 2
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding='valid'))


# Convolution Step 3
model.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))

# Convolution Step 4
model.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))


# Convolution Step 5
model.add(Convolution2D(256, 3, strides=(1,1), padding='same', activation = 'relu'))

# Max Pooling Step 3
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid'))

# Flattening Step --> 6*6*256 = 9216
model.add(Flatten())

# Full Connection Steps
# 1st Fully Connected Layer
model.add(Dense(units = 4096, activation = 'relu'))

# 2nd Fully Connected Layer
model.add(Dense(units = 4096, activation = 'relu'))

# 3rd Fully Connected Layer
model.add(Dense(units = 10, activation = 'softmax'))


print("[INFO]: Model Summary")
model.summary()

#######################################################################################

print("[INFO]: Initialize Adam Optimizers")

from keras.optimizers import Adam
import keras
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#######################################################################################

from keras.callbacks import CSVLogger

filepath="training_conf9.log"
path = F"/content/drive/uygulama/{filepath}" 
csv_logger = CSVLogger(path, separator=',', append=False)  

#######################################################################################

print("[INFO]: Training Model...")

train_num = train_data.n
valid_num = valid_data.n

train_batch_size = train_data.batch_size # choose 128
valid_batch_size = valid_data.batch_size #default 32

STEP_SIZE_TRAIN = train_num//train_batch_size #144
STEP_SIZE_VALID = valid_num//valid_batch_size #144

history = model.fit_generator(generator=train_data,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_data,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=[csv_logger],
                    epochs=25
)

#######################################################################################

print("[INFO]: Check training_conf9.log for model accuracy log ")
print("[INFO]: Saving Model and Weights ")
#saving model
filepath="model_conf1.hdf5"
path = F"/content/drive/uygulama/{filepath}" 
model.save(path)

filepath2="model_wieghts_conf1.h5"
path = F"/content/drive/uygulama/{filepath2}" 
model.save_weights(path)

###############################################################################

#plotting training values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

filepath_acc="conf1_accuracy.png"
path = F"/content/drive/uygulama/{filepath_acc}"
plt.savefig(path)

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

filepath_acc="conf1_loss.png"
path = F"/content/drive/uygulama/{filepath_acc}"
plt.savefig(path)

plt.show()

###############################################################################



































