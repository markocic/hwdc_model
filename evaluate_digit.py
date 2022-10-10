import imageio
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)



#gpus = tf.config.list_physical_devices('GPU')
#if gpus:
  # Restrict TensorFlow to only use the first GPU
#  try:
#    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
#    print(e)


# Loading a model

model = load_model("test_model.h5")

# Load image from a URL
im = imageio.imread("7.png")

# Convert RGB values to grayscale
gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
#plt.imshow(gray, cmap = plt.get_cmap('gray'))
#plt.show()


# reshaping and normalizing the image

img_rows, img_cols = 28, 28
gray = gray.reshape(1, img_rows, img_cols, 1)
gray = gray / 255

# Predict digit

prediction = model.predict(gray)
print(f"Broj sa slike je: {prediction.argmax()}")