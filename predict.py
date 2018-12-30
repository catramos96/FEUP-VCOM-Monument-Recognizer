#!/usr/bin/env python

from keras.applications.vgg16 import decode_predictions
import sys
import os
import cv2
import model_resources
import data_resources
import constants
from prepare_dataset import draw_image_box
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
from keras import optimizers
import prepare_dataset

ok = True

if len(sys.argv) > 2:
    image_path = sys.argv[1]
    model_path = sys.argv[2]

    if(not os.path.isfile(model_path) and (not os.path.isfile(image_path))):
        ok = False
else:
    ok = False

if(not ok):
    print("python predict.py [image_path] [model_path]")
    exit()

n_classes = len(data_resources.classes)

# export trained model
model = model_resources.create_model(
    constants.IMAGE_SIZE, n_classes+4)
model.load_weights(model_path)

# predict label
img = image.load_img(path=image_path, target_size=(
    constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
img = image.img_to_array(img)
test_img = img.reshape(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3)

predictions = model.predict(test_img)
print(predictions)

image = cv2.imread(image_path)
image = cv2.resize(image, (constants.IMAGE_SIZE, constants.IMAGE_SIZE))

draw_image_box(image, predictions[n_classes],
               predictions[n_classes+1], predictions[n_classes+2], predictions[n_classes+3])
