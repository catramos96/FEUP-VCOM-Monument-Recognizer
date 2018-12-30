#!/usr/bin/env python

import constants
import prepare_dataset
import data_resources
import model_resources
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint


from keras import backend as K
K.clear_session()

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
callbacks_list = [callback]

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

# save
model_resources.save_best_model()

# statistics
model_resources.plot_history(history)

K.clear_session()
