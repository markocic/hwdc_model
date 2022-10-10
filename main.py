import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)

# Loading MNIST dataset from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28

# Reshaping data to be ROWS x COLS x 1 since we have 784 neurons
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Since current values go from 0 to 255 we are going to divide them by 255 to get values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# Number of outputs (10 digits)
num_classes = 10

# to_categorical function is going to convert each number to a 10 x 1 vector
# where the index represents "solution" and it's the only value 1
# for example: 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#Designing a model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # This will randomly drop 25% of data to avoid overfitting
model.add(Flatten())  # Converts last hidden layer to 1D array
model.add(Dense(128, activation='relu'))  # Adding one more hidden layer
model.add(Dropout(0.5))  # This time it will randomly drop 50% of data

# The “softmax” activation is used when we’d like to classify the data into a number of pre-decided classes.
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

batch_size = 128  # A batch size is the number of training examples in one forward or backward pass.
epochs = 10  # An epoch is one forward pass and one backward pass of all training examples.


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("test_model.h5")

