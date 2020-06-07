#Colab için kod yapısı
########################################################################

#Mount Drive
from google.colab import drive
drive.mount('/content/drive')

########################################################################

#Instal Kaggle
!pip install -q kaggle

#Authentication Kaggle and show my api information 
!mkdir -p ~/.kaggle 

#'/content/drive/kaggle.json'  -->path of kaggle.json file 
!cp '/content/drive/kaggle.json' ~/.kaggle/ 

!chmod 600 ~/.kaggle/kaggle.json
!cat /root/.kaggle/kaggle.json

########################################################################

#Drive'daki bir klasöre gitme - Bu denenecek
import sys
sys.path.insert(0, 'drive/uygulama')

########################################################################

#Eğer kod hazırsa burada çalıştırılır
!python /content/drive/uygulama/confx.py

########################################################################

#Import primary library
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os 
import sys
import random

########################################################################

#Extract dataset
from zipfile import ZipFile
file_name = '/content/tomato.zip'

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print("[INFO]: Extracted dataset zip file")


########################################################################

#Import tensorflow and keras library
import tensorflow as tf
import keras_preprocessing
from tensorflow.keras.preprocessing import image
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras

print("[INFO]: Tensorflow version{}".format(tf.__version__))
state_gpu = tf.test.gpu_device_name()
print("[INFO]: GPU usage{0}".format(state_gpu))

########################################################################

# ImageDataGenerator and get train data and validation data
TRAINING_DIR = '/content/New Plant Diseases Dataset(Augmented)/train/' 
VALIDATION_DIR = '/content/New Plant Diseases Dataset(Augmented)/valid/'

########################################################################

rot_range = 40

# this is the augmentation configuration we will use for training
train_gen = ImageDataGenerator(
rescale = 1./255,
rotation_range=rot_range,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

valid_gen = ImageDataGenerator(rescale = 1./255)

########################################################################

TARGET_SIZE = (227,227)
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 32
SEED = 42

#Data Iterator
train_data = train_gen.flow_from_directory(
TRAINING_DIR,
target_size = TARGET_SIZE,
class_mode = 'categorical',
color_mode = "rgb",
batch_size = TRAIN_BATCH_SIZE,
shuffle = True,
seed = SEED
)

valid_data = valid_gen.flow_from_directory(
VALIDATION_DIR,
target_size = TARGET_SIZE,
class_mode = 'categorical',
color_mode = "rgb",
batch_size = VALID_BATCH_SIZE
)

########################################################################

#Show class indis and class name
for cl_indis, cl_name in enumerate(train_data.class_indices):
     print(cl_indis, cl_name)

########################################################################

# Initializing the CNN based AlexNet
model = Sequential()

#valid:zero padding, same:keep same dimensionality by add padding

# Convolution Step 1
model.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(227, 227, 3), activation = 'relu'))
model.add(BatchNormalization()) 

# Max Pooling Step 1
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid'))

# Convolution Step 2
model.add(Convolution2D(256, 5, strides = (1, 1), padding='same', activation = 'relu'))
model.add(BatchNormalization()) 

# Max Pooling Step 2
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding='valid'))


# Convolution Step 3
model.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))
model.add(BatchNormalization()) 

# Convolution Step 4
model.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))
model.add(BatchNormalization())

# Convolution Step 5
model.add(Convolution2D(256, 3, strides=(1,1), padding='same', activation = 'relu'))
model.add(BatchNormalization()) 

# Max Pooling Step 3
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid'))

# Flattening Step --> 6*6*256 = 9216
model.add(Flatten())

# Full Connection Steps
# 1st Fully Connected Layer
model.add(Dense(units = 4096, activation = 'relu'))
model.add(BatchNormalization()) 

# 2nd Fully Connected Layer
model.add(Dense(units = 4096, activation = 'relu'))
model.add(BatchNormalization()) 

# 3rd Fully Connected Layer
model.add(Dense(units = 10, activation = 'softmax'))

model.summary()

########################################################################

LEARNING_RATE = 0.0001
#LEARNING_RATE = 0.001

#Optimizer
opt = Adam(lr = LEARNING_RATE)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

########################################################################

from keras.callbacks import CSVLogger

#Log callback
filepath="training_confx.log"
path = F"storage/{filepath}" 
csv_logger = CSVLogger(path, separator=',', append=False)  

########################################################################

#Print Some Train Information
print("Train Batch Size: {0}\nValidation Batch Size: {1}\nTarget Size: {2}\nLearning Rate:{3}".format(TRAIN_BATCH_SIZE,VALID_BATCH_SIZE,TARGET_SIZE,LEARNING_RATE))

########################################################################

TRAINING_NUM = train_data.n #or train_data.samples
VALID_NUM = valid_data.n
EPOCHS = 25

STEP_SIZE_TRAIN = TRAINING_NUM // TRAIN_BATCH_SIZE 
STEP_SIZE_VALID = VALID_NUM // VALID_BATCH_SIZE

#Train Model
history = model.fit_generator(generator = train_data,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data = valid_data,
                    validation_steps = STEP_SIZE_VALID,
                    callbacks=[csv_logger],
                    epochs = EPOCHS
)

########################################################################

#Save Model
filepath="Model_confx.hdf5"
path = F"/content/drive/My Drive/uygulama/{filepath}" 
model.save(path)

filepath2="Model_wieghts_confx.h5"
path2 = F"/content/drive/My Drive/uygulama/{filepath2}" 
model.save_weights(path2)

########################################################################

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Plot Accuracy and Loss

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

plt.show()

########################################################################






# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



########################################################################



