#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author   : MuyunLi
Date     : 7/19/20 11:22 AM
FileName : switch_images.py
"""

from PIL import Image
import glob

target_path = "./target_images"
image_path = "images/*"

def resize_image(image_path, w, h):

    for image in glob.glob(image_path):
        img = Image.open(image)
        image_name= img.fp.name.split("/")[1]
        save_path = "{}/{}".format(target_path,image_name)
        img.resize((w, h), Image.ANTIALIAS).save(save_path, quality=95)


resize_image(image_path, 64, 64)

