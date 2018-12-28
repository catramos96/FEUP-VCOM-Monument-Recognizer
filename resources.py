from keras.applications import VGG16
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
from keras import callbacks
from keras.applications.vgg16 import VGG16

n_epochs = 50  # 50
validation_split = 0.3

batch_size = 15   # 32       #treinar com 20
image_size = 100  # 128     #treinar com 350

model_name = "monuments.h5"
weights_name = "weights.best.hdf5"
plot_name = "training_plot.png"

labels = ["arrabida", "camara", "clerigos", "musica", "serralves"]


def create_model():
    model = VGG16(weights='imagenet', include_top=False,
                  input_shape=(image_size, image_size, 3))

    for layer in model .layers[:-4]:
        layer.trainable = False

    flatten = layers.Flatten()
    output = layers.Dense(len(labels) + 4, activation='softmax')

    inp2 = model.input
    out2 = output(flatten(model.output))

    model = models.Model(inp2, out2)

    return model


def prepare_datasets(train_directory):

    # Step 1.3: Give random transformations to sets
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    #test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_directory,
        subset='training',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = train_datagen.flow_from_directory(
        train_directory,
        subset='validation',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, test_generator


def plot_history(history):
    loss_list = [s for s in history.history.keys(
    ) if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys()
                     if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys(
    ) if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys()
                    if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'r', label='Training loss (' +
                 str(str(format(history.history[l][-1], '.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'm', label='Validation loss (' +
                 str(str(format(history.history[l][-1], '.5f'))+')'))
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(
            format(history.history[l][-1], '.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'c', label='Validation accuracy (' + str(
            format(history.history[l][-1], '.5f'))+')')

    plt.title('Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(plot_name)
    plt.show()


def save_best_model():

    model = create_model(len(labels))
    model.load_weights(weights_name)
    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss='categorical_crossentropy',
        metrics=['acc'])

    model.save(model_name)


class CallbackSaveMode(callbacks.Callback):
    val_acc = 0.0
    val_loss = 1000
    acc = 0.0
    loss = 1000

    def on_epoch_end(self, batch, logs={}):
        new_val_acc = logs.get('val_acc')
        new_val_loss = logs.get('val_loss')
        new_acc = logs.get('acc')
        new_loss = logs.get('loss')

        # if(new_acc >= self.acc and new_loss <= self.loss and new_val_acc >= self.val_acc and new_val_loss <= self.val_loss):

        # save while loss of the training and validation set are decreasing
        if(new_loss <= self.loss and new_val_loss <= self.val_loss):
            self.val_acc = new_val_acc
            self.val_loss = new_val_loss
            self.acc = new_acc
            self.loss = new_loss
            self.model.save_weights(weights_name)
            print("Accuracy and Loss improved!")
        else:
            print("Nothing improved!")

        print("BEST\naccuracy {}\nloss {}\nvalidation accuracy {}\nvalidation loss {}".format(
            round(self.acc, 4), round(self.loss, 4), round(self.val_acc, 4), round(self.val_loss, 4)))
