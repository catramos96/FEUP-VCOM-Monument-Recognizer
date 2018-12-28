import resources
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from keras import backend as K
K.clear_session()

# model
model = resources.create_model()
model.compile(
    optimizer=optimizers.RMSprop(lr=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# data
train_set, test_set = resources.prepare_datasets(
    'data/images')

# train
callback = resources.CallbackSaveMode()
callbacks_list = [callback]

history = model.fit_generator(
    train_set,
    steps_per_epoch=20,  # 20,  # train_set.samples/train_set.batch_size,
    epochs=resources.n_epochs,
    validation_data=test_set,
    validation_steps=10,  # test_set.samples/test_set.batch_size,
    verbose=1,
    callbacks=callbacks_list)

# save
resources.save_best_model()

# statistics
resources.plot_history(history)

K.clear_session()
