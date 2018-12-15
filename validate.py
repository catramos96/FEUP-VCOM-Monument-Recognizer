import sys, os
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator



if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
	print("Error no image path specified")
	exit()


vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape = (64,64,3))

for layer in vgg_conv.layers[:-4]:
	layer.trainable = False
	
#Step 1.2: Create model
model = models.Sequential()
model.add(vgg_conv)

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

model.summary()
model.compile(
	optimizer=optimizers.RMSprop(lr=1e-4), 
	loss = 'categorical_crossentropy', 
	metrics = ['acc'])

model.load_weights("monuments.h5")


#Test image
img = image.load_img(path=image_path,target_size=(64,64,3))
img = image.img_to_array(img)
test_img = img
test_img = img.reshape(1,64,64,3)

#Get class
labels = ["Arrabida", "Camara", "Clerigos", "Musica", "Serralves"]

img_class = model.predict_classes(test_img)
classname = img_class[0]

#Get accuracy
img_class = model.predict(test_img)
prediction = img_class[0]
class_prediciton = prediction[classname] * 100

#print("Class number: ",classname)
print("Class:", labels[classname])
print("Accuracy: %6.2f%%" %class_prediciton)
print("Probabilities Array: ", prediction)