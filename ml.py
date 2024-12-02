import numpy as np
import tensorflow as tf
from keras import layers
from keras import models
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from keras.layers import Conv1D, Conv2D, Dropout, MaxPooling1D, Activation, LeakyReLU, SpatialDropout1D, GlobalAveragePooling1D, Flatten, Dense, Dropout,  MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, LSTM
import keras_tuner as kt
import data_pca

import matplotlib.pyplot as plt

import visualkeras

x_train = data_pca.x_train
y_train = data_pca.y_train
x_val = data_pca.x_val
y_val = data_pca.y_val # UNWSAPPED (i.e., correct)

model = Sequential()

model.add(Conv1D(25, 11, strides=2, padding='same', input_shape=(129, 5)))
model.add(Activation('selu'))

model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv1D(25, 11, strides=1, padding='same'))
model.add(Activation('selu'))

model.add(MaxPooling1D(pool_size=5, strides=3, padding='valid'))

model.add(Conv1D(50, 11, strides=1, padding='same'))
model.add(Activation('selu'))

model.add(SpatialDropout1D(0.5))
model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size=5, strides=3, padding='valid'))

model.add(Conv1D(100, 11, strides=1, padding='same'))
model.add(Activation('selu'))

model.add(SpatialDropout1D(0.5))
model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size=3, strides=3, padding='valid'))

model.add(Conv1D(100, 11, strides=1, padding='same'))
model.add(Activation('selu'))

model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))

model.add(Flatten())
 
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=.001), # SGD is low key good tho
              metrics=['accuracy'])

class_counts = np.bincount(np.argmax(y_train, axis=1))
total_samples = sum(class_counts)
class_weight_dict = {i: total_samples / count for i, count in enumerate(class_counts)}
'''class_weight_dict[3] += 1.0
class_weight_dict[4] += 3.0
class_weight_dict[0] -= 1.0'''
print(class_counts, total_samples, class_weight_dict)

checkpoint_filepath = '/Users/shaum/eeg-stuffs/checkpoints/model.keras'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
history = model.fit(x_train, y_train, batch_size=128, epochs=100, shuffle=True, validation_data=(x_val, y_val),
                    class_weight=class_weight_dict,
                    callbacks=[model_checkpoint_callback])

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
print(f'best accuracy: {max(val_acc_per_epoch)}')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

