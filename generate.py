#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author   : MuyunLi
Date     : 6/22/20 6:12 PM
FileName : generate.py

Use the generator model of DCGAN and the generator parameter file obtained by training to generate pictures
"""


import numpy as np
from PIL import Image

from network import *


save_path = "./BatchTest"

def generate():
    # Construction generator
    g = generator_model()

    # Configuration generator
    g.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1))

    # Load the trained generator parameters
    g.load_weights("generator_weight")

    # Continuous uniformly distributed random data (noise)
    random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

    # Using random data as input, the generator generates image data
    images = g.predict(random_data, verbose=1)

    # Use the generated image data to generate PNG images
    for i in range(BATCH_SIZE):
        image = images[i] * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save("{}/image-{}.png".format(save_path,i))


if __name__ == "__main__":
    generate()