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

# train
history = model.fit_generator(
    prepare_dataset.generator(constants.BATCH_SIZE, (constants.IMAGE_SIZE,
                                                     constants.IMAGE_SIZE), constants.VALIDATION_SPLIT, True),
    # 20,  # train_set.samples/train_set.batch_size,
    steps_per_epoch=constants.STEPS_PER_EPOCH,
    epochs=constants.N_EPOCHS,
    validation_data=prepare_dataset.generator(constants.BATCH_SIZE, (constants.IMAGE_SIZE,
                                                                     constants.IMAGE_SIZE), constants.VALIDATION_SPLIT, False),
    validation_steps=constants.VALIDATION_STEPS,
    verbose=1,
    callbacks=callbacks_list)

# save
resources.save_best_model()

# statistics
resources.plot_history(history)

K.clear_session()
