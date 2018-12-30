#!/usr/bin/env python

from keras import optimizers
from keras import callbacks
from keras import models
from keras import layers
from keras.applications.vgg16 import VGG16
import data_resources
import tensorflow as tf
from keras import backend as K
import numpy as np

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
    model.add(layers.Dense(output_size))

    model.compile(
        optimizer=optimizers.Adam(lr=1e-4),
        loss='mse',
        metrics=[Accuracy, IoU])

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


class CallbackSaveMode(callbacks.Callback):

    iou = -1
    accuracy = 0
    loss = -1

    val_iou = -1
    val_accuracy = 0
    val_loss = -1

    model_path = ""

    def set_model_path(self, model_path):
        self.model_path = model_path

    def on_epoch_end(self, batch, logs={}):
        new_iou = logs.get('IoU')
        new_accuracy = logs.get('Accuracy')
        new_val_iou = logs.get('val_IoU')
        new_val_accuracy = logs.get("val_Accuracy")
        new_loss = logs.get("loss")
        new_val_loss = logs.get("val_loss")

        if((new_loss <= self.loss and new_val_loss <= self.val_loss) or (self.loss == -1 and self.val_loss == -1)):
            iou = new_iou
            accuracy = new_accuracy
            loss = new_loss
            val_iou = new_val_iou
            val_accuracy = new_val_accuracy
            val_loss = new_val_loss

            self.model.save_weights(self.model_path)

            print("IMPROVEMENT")

        print("\nBEST_SAVED_MODEL (train : validation):\nIOU: \t\t{} : {}\nACCURACY: \t{} : {}\nLOSS: \t\t{} : {}\n".format(
            round(new_iou, 5), round(new_val_iou, 5), round(new_accuracy, 5), round(new_val_accuracy, 5), round(new_loss, 5), round(new_val_loss, 5)))


def calculate_accuracy(y_true, y_pred):

    results = []
    n_classes = len(data_resources.classes)

    for i in range(0, y_true.shape[0]):

        # set the types so we are sure what type we are using
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)

        max_ = 0
        indx = -1

        j = 0

        # check the predicted class
        for j in range(n_classes):
            if(y_pred[i, j] >= max_):
                max_ = y_pred[i, j]
                indx = j

        r = y_true[i, indx].astype(np.float32)

        results.append(r)

    accuracy = np.mean(results)

    return accuracy


def calculate_iou(y_true, y_pred):

    results = []
    n_classes = len(data_resources.classes)

    for i in range(0, y_true.shape[0]):

        # set the types so we are sure what type we are using
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)

        # boxTrue
        true_x_min = y_true[i, 0 + n_classes]
        true_y_min = y_true[i, 1 + n_classes]
        true_width = y_true[i, 2 + n_classes]
        true_height = y_true[i, 3 + n_classes]
        true_x_max = true_x_min + true_width
        true_y_max = true_y_min + true_height
        true_area = true_height * true_width

        # boxPred
        pred_x_min = y_pred[i, 0 + n_classes]
        pred_y_min = y_pred[i, 1 + n_classes]
        pred_width = y_pred[i, 2 + n_classes]
        pred_height = y_pred[i, 3 + n_classes]
        pred_x_max = pred_x_min + pred_width
        pred_y_max = pred_y_min + pred_height
        pred_area = pred_height * pred_width

        # calculate coordinated of the intersection box
        int_x_min = np.max([true_x_min, pred_x_min])
        int_y_min = np.max([true_y_min, pred_y_min])
        int_x_max = np.min([true_x_max, pred_x_max])
        int_y_max = np.min([true_y_max, pred_y_max])
        int_width = int_x_max - int_x_min
        int_height = int_y_max - int_y_min
        int_area = int_width * int_height

        # no intersection
        if(int_area < 0):
            int_area = 0

        # iou = intersection / union
        iou = int_area / ((true_area + pred_area) - int_area)

        iou = iou.astype(np.float32)

        results.append(iou)

    # return the mean IoU score for the batch
    return np.mean(results)


def IoU(y_true, y_pred):

    return tf.py_func(calculate_iou, [y_true, y_pred], tf.float32)


def Accuracy(y_true, y_pred):
    return tf.py_func(calculate_accuracy, [y_true, y_pred], tf.float32)
