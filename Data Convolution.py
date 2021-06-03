import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import sklearn.metrics
import pandas as pd
import random
import glob
import PIL
from PIL import Image as PImage

import tensorflow as tf
import keras
import tensorflow.keras as krs
import tensorflow.keras.layers as layers
import tensorflow.keras.layers.experimental.preprocessing as prepro
import tensorflow.compat.v1 as tf1

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

tf1.Session(config=tf1.ConfigProto(log_device_placement=True))

IDcolumns = ['database_id','img_name']
DDcolumns = ['Galaxy ID','Disc To Total', 'Kappa Co Rot']

ImageData = pd.read_csv("E:/PythonProjects/UniProject/Data/GZcomposite/manifest.csv")
DiscData = pd.read_csv("E:/PythonProjects/UniProject/Data/MyDB")

ImageData = (ImageData.loc[ImageData['simulation'] == 'Reference model L100N1504 simulation'])[IDcolumns]

IndexList = []
for GID in DiscData['Galaxy ID']:
   if np.array(ImageData['database_id']).__contains__(GID):
       IndexList.append(GID)

DiscData= (DiscData[DiscData['Galaxy ID'].isin(IndexList)].reset_index())[DDcolumns]

DiscData.insert(3,'Image Name',ImageData['img_name'],True)

train, test = train_test_split(DiscData,train_size=0.8,test_size=0.2,random_state=56)

train = (pd.DataFrame(train,columns=['Galaxy ID','Disc To Total', 'Kappa Co Rot','Image Name']).reset_index())[['Galaxy ID','Disc To Total', 'Kappa Co Rot','Image Name']]
test = (pd.DataFrame(test,columns=['Galaxy ID','Disc To Total', 'Kappa Co Rot','Image Name']).reset_index())[['Galaxy ID','Disc To Total', 'Kappa Co Rot','Image Name']]

print(train)
print(test)

trainImages = np.array(train['Image Name'])
trainImages = pd.DataFrame(trainImages,columns=['Image'])
testImages = np.array(test['Image Name'])
testImages = pd.DataFrame(testImages,columns=['Image'])

img_Train = []
for img in trainImages['Image']:
    img = PImage.open('E:/PythonProjects/UniProject/Data/GZcomposite/' + img)
    img = np.asarray(img)
    img_Train.append(img)
img_Train = np.array(img_Train)

img_Test = []
for img in testImages['Image']:
    img = PImage.open('E:/PythonProjects/UniProject/Data/GZcomposite/'+img)
    img= np.asarray(img)
    img_Test.append(img)
img_Test = np.array(img_Test)

trainImages = np.array(img_Train)
testImages = np.array(img_Test)

outDataTrain = np.array(train[['Disc To Total', 'Kappa Co Rot']])
outDataTest = np.array(test[['Disc To Total', 'Kappa Co Rot']])

print(np.shape(trainImages))
print(np.shape(outDataTrain))

batch_size =3
epochs = 2
learning_rate = 0.00001

model = Sequential()
model.add(Dense(3,activation='relu',input_shape=(424,424,3)))
model.add(Conv2D(4900, kernel_size=(20,20),activation='relu',padding='same',strides=(6)))
model.add(tf.keras.layers.LeakyReLU(0.1))
model.add(Dense(4900,activation='relu'))
model.add(Conv2D(1225,kernel_size=(5,5),activation='relu',padding='same',strides=(2)))
model.add(tf.keras.layers.Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1225,activation='relu'))
model.add(Conv2D(324,(5,5),activation='relu',padding='same',strides=(2)))
model.add(tf.keras.layers.Activation('relu'))
model.add(Dense(324,activation='relu'))
model.add(Conv2D(64,(5,5),activation='relu',padding='same',strides=(2)))
model.add(tf.keras.layers.LeakyReLU(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16,(4,4),activation='relu',padding='same',strides=(2)))
model.add(tf.keras.layers.Activation('relu'))
model.add(Dense(16,activation='relu'))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(tf.keras.layers.Activation('relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(2))

model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.optimizers.Adam(learning_rate=learning_rate),metrics=['accuracy'])
model.summary()
train = model.fit(img_Train,outDataTrain, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(img_Test,outDataTest))

tf.saved_model.save(model,'E:/PythonProjects/UniProject/Data/Model/')

accuracy = train.history['accuracy']
val_accuracy = train.history['val_accuracy']
loss = train.history['loss']
val_loss = train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()