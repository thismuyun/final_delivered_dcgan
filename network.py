#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author   : MuyunLi
Date     : 6/22/20 6:13 PM
FileName : network.py

DCGAN Deep convolution generation adversarial network
"""


import tensorflow as tf

# Hyper parameter
EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA_1 = 0.5

# Define the discriminator model
def discriminator_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(
        64, # 64 filters with a depth of 64
        (5, 5),  # The size of the filter in two dimensions is (5 * 5)
        padding='same',  # same Indicates that the size of the output is unchanged, so it is necessary to fill the periphery with 2 circles
        input_shape=(64, 64, 3)  # Enter the shape [64, 64, 3]. 3 means RGB primary colors   1 menas Grayscale
    ))
    model.add(tf.keras.layers.Activation("tanh"))  # Add Tanh activation layer
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))  # Pooling layer ,width and height are reduced by 2 times
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())   # Flat
    model.add(tf.keras.layers.Dense(1024))  # Fully connected layer of 1024  neurons
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Dense(1))  # 1 neuron fully connected layer
    model.add(tf.keras.layers.Activation("sigmoid"))  # Add Sigmoid activation layer

    return model

# Define the generator model
# Generate pictures from random numbers
def generator_model():
    model = tf.keras.models.Sequential()
    # The input dimension is 100, and the output dimension (number of neurons) is 1024 in the fully connected layer
    model.add(tf.keras.layers.Dense(input_dim=100, units=1024))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Dense(128 * 8 * 8))  # 8192 fully connected layers of neurons
    model.add(tf.keras.layers.BatchNormalization())   # Batch standardization
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Reshape((8, 8, 128), input_shape=(128 * 8 * 8, )))   # 8 x 8 Pixel
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(3, (5, 5), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))

    return model


# Construct a Sequential object, including a generator and a discriminator
# Input -> generator -> discriminator -> output
def generator_containing_discriminator(generator, discriminator):
    model = tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable = False  # Initially the discriminator cannot be trained
    model.add(discriminator)
    return model