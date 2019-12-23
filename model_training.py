import os 
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

device_name = tf.test.gpu_device_name()
print(device_name)
if device_name != '/device:GPU:0':
    device_name="/CPU:0"
print('device is : {}'.format(device_name))
data=np.load("shots.npz")

shots=data["shots"]
labels=data["labels"]
keys=data["keys"]
vals=data["vals"]
pd.DataFrame({"Video Id":keys,"Frames end at":vals})

validation_index=np.concatenate((np.arange(5202,5642),np.arange(6478,6830),np.arange(7182,7622),np.arange(7959,8303)))
# set up a mask to subset the training set
mask = np.ones(len(labels), np.bool)
mask[validation_index] = 0

# seperating validation set
x_tr=shots[mask]
x_te=shots[validation_index]
y_tr=labels[mask]
y_te=labels[validation_index]
# Image augmentation
train_image_generator = ImageDataGenerator(rescale=1./4096, 
                                           rotation_range=30,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=10,
                                           zoom_range=0.1,
                                           horizontal_flip=True,
                                          )
validation_image_generator = ImageDataGenerator(rescale=1./4096)
train_data_gen = train_image_generator.flow(x_tr,y_tr)
val_data_gen = validation_image_generator.flow(x_te,y_te)
IMG_HEIGHT = 128
IMG_WIDTH = 128
with tf.device(device_name):
    model = Sequential([
        Conv2D(16, 11, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,1)),
        MaxPooling2D(),
        Conv2D(32, 11, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
total_train=int(x_tr.shape[0])
total_val=int(x_te.shape[0])
epochs = 20
batch_size=128
#with tf.device('/device:GPU:0'):
history=model.fit_generator(
              train_data_gen,
              steps_per_epoch=total_train // batch_size,
              epochs=epochs,
              validation_data=val_data_gen,
              
              validation_steps=total_val // batch_size
          )
print("model trained")
print('\n# Evaluate on validation set')
results = model.evaluate(x_te, y_te, batch_size=batch_size)
print('test loss, test acc:', results)
model.save("trained_model.h5")
print("Model trained and saved in \"trained_model.h5\" ")