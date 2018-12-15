from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

'''
Phase 1: Model Creation
'''

def plot_history(history):
	loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
	val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
	acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
	val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

	if len(loss_list) == 0:
		print('Loss is missing in history')
		return 

	## As loss always exists
	epochs = range(1,len(history.history[loss_list[0]]) + 1)

	## Loss
	plt.figure(1)
	for l in loss_list:
		plt.plot(epochs, history.history[l], 'r', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
	for l in val_loss_list:
		plt.plot(epochs, history.history[l], 'm', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
	for l in acc_list:
		plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
	for l in val_acc_list:    
		plt.plot(epochs, history.history[l], 'c', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

	plt.title('Loss and Accuracy')
	plt.xlabel('Epochs')
	plt.legend()
	plt.show()

#Step 1.1: Import the pre trained model
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

#Step 1.3: Give random transformations to sets
train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')
	
test_datagen = ImageDataGenerator(rescale=1./255)


#Step 1.4: Create the datasets
training_set = train_datagen.flow_from_directory(
	'data/train',
	target_size=(64,64),
	batch_size=32,
	class_mode='categorical')
	
test_set = test_datagen.flow_from_directory(
	'data/validation',
	target_size=(64,64),
	batch_size=32,
	class_mode='categorical',
	shuffle=False)
	
#Step 1.5: Compiling the model
model.compile(
	optimizer=optimizers.RMSprop(lr=1e-4), 
	loss = 'categorical_crossentropy', 
	metrics = ['acc'])
	
#Step 1.6: Train the model
history = model.fit_generator(
	training_set,
	steps_per_epoch=training_set.samples/training_set.batch_size ,
	epochs=30,
	validation_data=test_set,
	validation_steps=test_set.samples/test_set.batch_size,
	verbose=1)
	
#Step 1.7: Plot History
plot_history(history)

#Step 1.7: Save the model
model.save('monuments.h5')
	