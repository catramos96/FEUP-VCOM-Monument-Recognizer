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

# parse predicions
classes_predictions = predictions[0][:len(data_resources.classes)]  # class
box_prediction = predictions[0][len(
    data_resources.classes):]       # normalized box

prepare_dataset.desnormalize_box(
    box_prediction, constants.IMAGE_SIZE)    # desnormalized box

# calculate best class
class_predicted = -1
max_ = 0

for i in range(len(classes_predictions)):

    if classes_predictions[i] > max_:
        max_ = classes_predictions[i]
        class_predicted = data_resources.classes[i]

print("Predicted class: {}".format(class_predicted))
print("Predicted bounding box: {}".format(box_prediction))

# draw image and box
image = cv2.imread(image_path)
image = cv2.resize(image, (constants.IMAGE_SIZE, constants.IMAGE_SIZE))

draw_image_box(
    image, box_prediction[0], box_prediction[1], box_prediction[2], box_prediction[3])
