from keras.applications.vgg16 import decode_predictions
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
    metrics=['accuracy'])

model.load_weights("weights.best.hdf5")

# predict label
img = image.load_img(path=image_path, target_size=(
    resources.image_size, resources.image_size, 3))
img = image.img_to_array(img)
test_img = img.reshape(1, resources.image_size, resources.image_size, 3)

predictions = model.predict(test_img)
print(predictions)
'''
# convert the probabilities to class labels
label = decode_predictions(predictions)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))
'''
'''
img_class = model.predict([test_img])
classname = img_class[0]

print(img_class)

  
prediction = img_class[0]

print(img_class)

#class_prediciton = prediction[classname] * 100
'''
'''
#print("Class number: ",classname)
print("Class:", resources.labels[classname])
print("Accuracy: %6.2f%%" % class_prediciton)
print("Probabilities Array: ", prediction)
'''
