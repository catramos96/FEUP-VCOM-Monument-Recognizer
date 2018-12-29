#!/usr/bin/env python

from keras import optimizers
from keras import callbacks
from keras import models
from keras import layers
from keras.applications.vgg16 import VGG16

model_name = "monuments.h5"
weights_name = "weights.best.hdf5"
plot_name = "training_plot.png"


def create_model(input_size, output_size):
    # Step 1.1: Import the pre trained model
    vgg_conv = VGG16(weights='imagenet', include_top=False,
                     input_shape=(input_size, input_size, 3))

    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    # Step 1.2: Create model
    model = models.Sequential()
    model.add(vgg_conv)
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_size, activation='softmax'))

    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    return model


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
    val_acc = -1
    val_loss = -1
    acc = -1
    loss = -1

    def on_epoch_end(self, batch, logs={}):
        new_val_acc = logs.get('val_acc')
        new_val_loss = logs.get('val_loss')
        new_acc = logs.get('acc')
        new_loss = logs.get('loss')

        # save while loss of the training and validation set are decreasing
        if((new_loss <= self.loss and new_val_loss <= self.val_loss) or (self.loss == -1 and self.val_loss == -1)):
            self.val_acc = new_val_acc
            self.val_loss = new_val_loss
            self.acc = new_acc
            self.loss = new_loss
            self.model.save_weights(weights_name)
            print("IMPROVEMENT!!!!!!!!!!!!!!!!!!!!!!!!!")

        print("BEST\naccuracy {}\nloss {}\nvalidation accuracy {}\nvalidation loss {}".format(
            round(self.acc, 4), round(self.loss, 4), round(self.val_acc, 4), round(self.val_loss, 4)))
