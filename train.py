import resources
from keras.callbacks import ModelCheckpoint
from keras import optimizers

# model
model = resources.create_model(len(resources.labels))

model.compile(
    optimizer=optimizers.RMSprop(lr=1e-4),
    loss='categorical_crossentropy',
    metrics=['acc'])

# data
train_set, test_set = resources.prepare_datasets(
    'data/train')

# train
callback = resources.CallbackSaveMode()
callbacks_list = [callback]

history = model.fit_generator(
    train_set,
    steps_per_epoch=train_set.samples/train_set.batch_size,
    epochs=resources.n_epochs,
    validation_data=test_set,
    validation_steps=test_set.samples/test_set.batch_size,
    verbose=1,
    callbacks=callbacks_list)

# save
resources.save_best_model()

# statistics
resources.plot_history(history)
