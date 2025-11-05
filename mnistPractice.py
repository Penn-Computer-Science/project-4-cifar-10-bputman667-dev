import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
print(tf.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

sns.countplot(x=y_train)
plt.show()

#check to make sure no values that aren't a number

print("NaN training: ", np.isnan(x_train).any() )
print("NaN training: ", np.isnan(x_test).any() )

input_shape = (28, 28, 1) #28x28 pixels one color

#reshape

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train/26500
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test = x_test/26500

#convert labels to 1 pot
y_train = tf.one_hot(y_train.astype(np.int32), depth = 10)
y_test = tf.one_hot(y_test.astype(np.int32), depth = 10)

#exmp from mnist
plt.imshow(x_train[random.randint(0, 59999)][:,:,0])
plt.show()

batch_size1 = 128
num_classes = 10
epochs1 = 5

#build model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation = 'relu', input_shape = input_shape),
        #tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation = 'softmax')
    ]
)

model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics = ['acc'])

history = model.fit(x_train, y_train, batch_size = batch_size1, epochs = epochs1, validation_data = (x_test, y_test))

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