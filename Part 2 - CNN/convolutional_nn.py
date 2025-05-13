import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras._tf_keras.keras.preprocessing import image

print(tf.__version__)
print(keras.__version__)

# Data preprocessing and augmentation

shape = 64
generators_params = {
    "target_size": (shape,shape),
    "batch_size": 32,
    "class_mode": 'binary',
}

train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    **generators_params
)

test_datagen = image.ImageDataGenerator(rescale=1./255)
test_set = train_datagen.flow_from_directory(
    'dataset/test_set',
    **generators_params
)

# CNN creation
model_file = 'model.keras'

if not os.path.isfile(model_file):
    cnn = keras.Sequential([
        keras.layers.Input((shape,shape,3)),
        keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

    # CNN training

    cnn.fit(x=train_set, validation_data=test_set, epochs=25)
    cnn.save(model_file)
else:
    cnn = keras.models.load_model(model_file)

# Testing on single predictions

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(shape, shape))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

test_pred = cnn.predict(test_image)
if test_pred[0][0] == 1:
    pred = 'dog'
else:
    pred = 'cat'
print(f'#1: {pred}')

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(shape, shape))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

test_pred = cnn.predict(test_image)
if test_pred[0][0] == 1:
    pred = 'dog'
else:
    pred = 'cat'
print(f'#2: {pred}')
