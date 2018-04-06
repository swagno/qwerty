#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:02:51 2018

@author: michal
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.models import load_model

import os
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
classifier = Sequential()
classifier = load_model('my_model.h5')  # creates a HDF5 file 'my_model.h5' = load_model('my_model.h5')

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)