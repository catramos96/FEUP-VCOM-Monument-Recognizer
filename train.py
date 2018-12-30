#!/usr/bin/env python

import constants
import prepare_dataset
import data_resources
import model_resources
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import os.path
import sys

model_path = ""
confif = ""
ok = True

if len(sys.argv) >= 3:
    config = sys.argv[1]
    model_path = sys.argv[2]

    if(config != "-o" and config != "-n" and (not os.path.isfile(model_path) and config == "-n")):
        ok = False
else:
    ok = False


if(not ok):
    print(
        "python train.py [-o|-n] [model_path]\n-o\ttrain old model\n-n\ttrain new model")
    exit()

'''
tensorboard --logdir=tensorboard  --port=8008
'''

# data
'''
X_training, X_validation, Y_training, Y_validation = prepare_dataset.prepare_datasets(
    constants.BATCH_SIZE, constants.IMAGE_SIZE, constants.VALIDATION_SPLIT)
'''

# model
model = model_resources.create_model(
    constants.IMAGE_SIZE, len(data_resources.classes)+4)

if(config == "-o"):
    model.load_weights(model_path)

# callbacks
tb = TensorBoard(
    log_dir=constants.TENSORBOARD_DIR,
    histogram_freq=1,
    batch_size=constants.BATCH_SIZE,
    write_graph=True,
    write_grads=True,
    write_images=True,
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None)

callback = model_resources.CallbackSaveMode()
callback.set_model_path(model_path)
callbacks_list = [callback]  # ,tb]

n_train, n_val = prepare_dataset.get_train_val_n_samples(
    constants.VALIDATION_SPLIT)

# train
is_balanced_data = True
batch_mult = 1

# adjust batch size for steps
if(is_balanced_data):
    batch_mult = 5

history = model.fit_generator(
    prepare_dataset.generator(constants.BATCH_SIZE, (constants.IMAGE_SIZE,
                                                     constants.IMAGE_SIZE), constants.VALIDATION_SPLIT, True, is_balanced_data),
    steps_per_epoch=max(
        1, n_train/(constants.BATCH_SIZE*batch_mult)),
    epochs=constants.N_EPOCHS,
    validation_data=prepare_dataset.generator(constants.BATCH_SIZE, (constants.IMAGE_SIZE,
                                                                     constants.IMAGE_SIZE), constants.VALIDATION_SPLIT, False, is_balanced_data),
    validation_steps=max(
        1, n_val/(constants.BATCH_SIZE*batch_mult)),
    verbose=1,
    callbacks=callbacks_list)

# statistics
model_resources.plot_history(history)
