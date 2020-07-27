#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author   : MuyunLi
Date     : 6/23/20 5:21 PM
FileName : processing_image.py
"""
import cv2
import os
import glob
import numpy as np
from scipy import misc
from PIL import Image
import sys

from network import *


input_data = sys.argv[1]

save_path = "./SingleTest"


def run(input_data):
    # img2 = cv2.imread(input_data)
    # if img2 is None:
    #     print('fail to load image!')
    #
    # img2 = cv2.imread(input_data,cv2.IMREAD_GRAYSCALE)
    input_data = misc.imread(input_data)  # imread uses PIL to read image data
    print("input_data.dtype : {}".format(input_data.dtype))
    print("input_data.shape : {}".format(input_data.shape))
    print(input_data)

    input_data_list = np.array(input_data)
    print("input_data_list.dtype : {}".format(input_data_list.dtype))
    print("input_data_list.shape : {}".format(input_data_list.shape))
    print(input_data_list)


    # Normalize the data to the value of [-1, 1], which is also the output range of the Tanh activation function
    img2 = (input_data_list.astype(np.float32) - 127.5) / 127.5

    print("img2.dtype : {}".format(img2.dtype))
    print("img2.shape : {}".format(img2.shape))
    print(img2)

    one_dim_data = img2.ravel()
    print("one_dim_data.dtype : {}".format(one_dim_data.dtype))
    print("one_dim_data.shape : {}".format(one_dim_data.shape))
    print(one_dim_data)


    one_dim_data_1 = one_dim_data[:128*100]
    print("one_dim_data_1.dtype : {}".format(one_dim_data_1.dtype))
    print("one_dim_data_1.shape : {}".format(one_dim_data_1.shape))
    print(one_dim_data_1)

    re_data = one_dim_data_1.reshape((128,100))
    print("re_data.dtype : {}".format(one_dim_data_1.dtype))
    print("re_data.shape : {}".format(one_dim_data_1.shape))
    # print(re_data)
    g = generator_model()

    # Configuration generator
    g.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1))

    # Load the trained generator parameters
    g.load_weights("generator_weight")

    random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
    print("random_data.dtype : {}".format(random_data.dtype))
    print("random_data.shape : {}".format(random_data.shape))
    # print(random_data)

    output_images = g.predict(re_data, verbose=1)
    print("output_images.dtype : {}".format(output_images.dtype))
    print("output_images.shape : {}".format(output_images.shape))
    n = 0
    # for i in range(0,128):
    #     image = output_images[i] * 127.5 + 127.5
    #     Image.fromarray(image.astype(np.uint8)).save("{}/image-{}.png".format(save_path, i))


    image = output_images[0] * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save("{}/image-{}.png".format(save_path, 127))


if __name__ == "__main__":
    run(input_data)
