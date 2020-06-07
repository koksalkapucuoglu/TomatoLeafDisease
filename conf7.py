import os # accessing directory structure
import sys
import random
import json
import zipfile

print("[INFO]: Imported primary library")

from zipfile import ZipFile
file_name = '/content/drive/kaggle/tomato_dataset1/tomato.zip'

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print("[INFO]: Extracted dataset zip file")

import tensorflow as tf
#tf.test.gpu_device_name()
print("[INFO]: GPU usage{0}".format(tf.test.gpu_device_name()))

#import tensorflow and kaggle library
import tensorflow as tf
import keras_preprocessing
from tensorflow.keras.preprocessing import image
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
#tf.__version__
# Any results you write to the current directory are saved as output.

print("[INFO]: Imported keras and tensorflow library")
print("[INFO]: Tensorflow version{}".format(tf.__version__))

# ImageDataGenerator and get train data and validation data
TRAINING_DIR = '/content/New Plant Diseases Dataset(Augmented)/train/' 
VALIDATION_DIR = '/content/New Plant Diseases Dataset(Augmented)/valid/'

print("[INFO]: Training DIR{}".format(TRAINING_DIR))
print("[INFO]: Validation DIR{}".format(VALIDATION_DIR))

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
 
#rgb --> the images will be converted to have 3 channels.


print("[INFO]: Training Batch Size:",32)
print("[INFO]: Validation Batch Size:",32)

train_data = train_gen.flow_from_directory(
TRAINING_DIR,
target_size=(227,227),
class_mode='categorical',
color_mode="rgb",
batch_size=32
)

valid_data = valid_gen.flow_from_directory(
VALIDATION_DIR,
target_size=(227,227),
class_mode='categorical',
color_mode="rgb"
)


print("[INFO]: Classes of Dataset")

for cl_indis, cl_name in enumerate(train_data.class_indices):
     print(cl_indis, cl_name)

# import keras libary for create cnn layer 
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.layers.normalization import BatchNormalization
print("[INFO]: Imported CNN library")

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

print("[INFO]: Initialize Adam Optimizers")

from keras.optimizers import Adam
import keras
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

print("[INFO]: Training Model")


#train_num/128 = 144
#valid_num//128 = 35
train_num = train_data.n
valid_num = valid_data.n

train_batch_size = train_data.batch_size # choose 64
valid_batch_size = valid_data.batch_size #default 32

STEP_SIZE_TRAIN = train_num//train_batch_size 
STEP_SIZE_VALID = valid_num//valid_batch_size 

print("[INFO]: Step Size Train:{}".format(STEP_SIZE_TRAIN))
print("[INFO]: Step Size Validation:{}".format(STEP_SIZE_VALID))

history = model.fit_generator(generator=train_data,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_data,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=25
)

#saving model
filepath="AlexNetModel_conf7.hdf5"
path = F"/content/drive/uygulama/{filepath}" 
model.save(path)

filepath2="model_wieghts_conf7.h5"
path = F"/content/drive/uygulama/{filepath2}" 
model.save_weights(path)

filepath3="model_keras_conf7.h5"
path = F"/content/drive/uygulama/{filepath3}" 
model.save_weights(path)

###############################################################################3

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

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

filepath_acc="conf6_accuracy.png"
path = F"/content/drive/uygulama/{filepath_acc}"
plt.savefig(path)

plt.show()





































