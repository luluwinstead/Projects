# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:31:09 2019
@author: apblossom
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

import datetime
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print ("train_images.shape",train_images.shape)
print ("len(train_labels)",len(train_labels))
print("train_labels",train_labels)

print("test_images.shape", test_images.shape)
print("len(test_labels)", len(test_labels))
print("test_labels", test_labels)

# Start the timer for run time
start_time = datetime.datetime.now()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
image = test_images[171]
image = image.reshape((28 ,28))
plt.imshow(image)
plt.show()

#Reshape to 60000 x 784
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

# Reduce range of data to 0,1
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# `to_categorical` converts this into a matrix with as many
# columns as there are classes. The number of rows
# stays the same.
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Create the feedforward  netowrk
network = models.Sequential()
#Add the first hidden layer specifying the input shape
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
#Add a second hidden layer
network.add(layers.Dense(512, activation='relu'))
#Add the output layer
network.add(layers.Dense(10, activation='softmax'))
#compile the model
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
#print a summary of the model
network.summary()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fit Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train the model
network.fit(train_images, train_labels, epochs=5, batch_size=64, verbose=1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#calculate and print the loss statistics
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)

#calculate and print the tome to run
stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)
