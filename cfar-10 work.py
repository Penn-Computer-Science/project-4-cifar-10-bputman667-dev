import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
print(tf.__version__)

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



#check to make sure no values that aren't a number



input_shape = (32, 32, 3) #28x28 pixels one color

#reshape

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#convert labels to 1 pot
from keras.utils import to_categorical

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

batch_size1 = 256
num_classes = 10

#build model
counter = 0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation = 'relu', input_shape = input_shape),
        #tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation = 'relu', input_shape = input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation = 'relu', input_shape = input_shape),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation = 'relu', input_shape = input_shape),
        tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation = 'softmax'),
    ],
)

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc']) #removed sparse from categorical crossentropy

history = model.fit(x_train, y_train,epochs=10, validation_split=0.1)
#plot out training and validation accuracy and loss

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color = 'b', label = 'training loss')
ax[0].plot(history.history['val_loss'], color = 'r', label = 'validation loss')
legend = ax[0].legend(loc='best', shadow = True)
ax[0].set_title('loss')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')


ax[1].plot(history.history['acc'], color = 'b', label = 'training accuracy')
ax[1].plot(history.history['val_acc'], color = 'r', label = 'validation accuracy')
legend = ax[1].legend(loc='best', shadow = True)
ax[1].set_title('accuracy')
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')

plt.tight_layout()
plt.show()