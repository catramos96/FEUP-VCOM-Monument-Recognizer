import sys
import os
import resources
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
from keras import optimizers


if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    print("Error no image path specified")
    exit()

# export trained model
model = resources.create_model(len(resources.labels))

model.summary()
model.compile(
    optimizer=optimizers.RMSprop(lr=1e-4),
    loss='categorical_crossentropy',
    metrics=['acc'])

model.load_weights("monuments.h5")

# predict label
img = image.load_img(path=image_path, target_size=(64, 64, 3))
img = image.img_to_array(img)
test_img = img
test_img = img.reshape(1, 64, 64, 3)


img_class = model.predict_classes(test_img)
classname = img_class[0]

# confidence
img_class = model.predict(test_img)
prediction = img_class[0]
class_prediciton = prediction[classname] * 100

#print("Class number: ",classname)
print("Class:", resources.labels[classname])
print("Accuracy: %6.2f%%" % class_prediciton)
print("Probabilities Array: ", prediction)
