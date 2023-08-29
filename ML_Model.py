#importing those required---->

import os

import cv2

import numpy as np

import tensorflow as tf

import sklearn

import livelossplot

import matplotlib.pyplot as plt

from tensorflow import keras

from sklearn.model_selection import train_test_split 

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense

from keras.utils import to_categorical


#creating model's required data---->

NUM_CLASSES = 10
BATCH_SIZE = 85
EPOCHS =25


# using liveloseplot for live graph performance ---->

plot_losses = livelossplot.PlotLossesKeras()

NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS =25

# creating loadimages function--->

def loadImages(path, urls, target, desired_size):
    images=[]
    labels=[]
    
    for url in urls:
        img_path = os.path.join(path, url)
        img = cv2.imread(img_path)
        img = cv2.resize(img, desired_size)
        images.append(img)
        labels.append(target)
        
    return  images, labels


#Giving path to model---->
#path of pneumonia images :
    
pneu_path= 'C:\\Users\\codew\\kaggle datasets\\chest_xray\\test\\PNEUMONIA'

pneu_urls=  os.listdir(pneu_path)

pneuImages , pneuTargets = loadImages(pneu_path, pneu_urls, 1, (156, 156))

#path of normal images :

nor_path= 'C:\\Users\\codew\\kaggle datasets\\chest_xray\\test\\NORMAL'

nor_urls=  os.listdir(nor_path)

norImages , norTargets=loadImages(nor_path, nor_urls, 0, (156, 156))


#convert target and imagses in array beacuse of list form of return images and labels as normal and pneumonia separately--->

data = np.r_[pneuImages, norImages]
targets = np.r_[pneuTargets, norTargets]
data = data/255.0


# load data & create train_test ---->

x_train,x_test,y_train,y_test = train_test_split(data, targets,test_size=0.3)


# Categorical encode labels--->

y_train = to_categorical(y_train, NUM_CLASSES)

y_test = to_categorical(y_test, NUM_CLASSES)


# creating model---->

model = Sequential()

# Adjust the input shape to match your image size
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(156, 156, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=18, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(units=512, activation='relu'))

model.add(Dropout(0.6))

model.add(Dense(units=212, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=156, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(units=156, activation='relu'))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))


# Compile the model--->

learning_rate=0.1

adam = keras.optimizers.Adam(learning_rate)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# train model--->

model.fit(x_train, y_train,
          batch_size = BATCH_SIZE,
          epochs = EPOCHS,
          verbose = 1,
          callbacks = [plot_losses],
          validation_data = (x_test, y_test))


score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Now saving my model---->

#model.save('model.h5')