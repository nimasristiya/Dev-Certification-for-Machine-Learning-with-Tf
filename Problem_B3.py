# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.83 and logs.get('val_accuracy') > 0.83):
            self.model.stop_training = True


def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

    TRAINING_DIR = "data/rps/"
    training_datagen = ImageDataGenerator(rescale=1. / 255,
                                          rotation_range=0.2,
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          fill_mode='nearest',
                                          validation_split=0.1
                                          )

    val_datagen = ImageDataGenerator(rescale=1. / 255,
                                     validation_split=0.1
                                     )
    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    train_generator = training_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                           target_size=(150, 150),
                                                           color_mode='rgb',
                                                           class_mode='categorical',
                                                           shuffle=True,
                                                           batch_size=32,
                                                           seed=123,
                                                           subset='training'
                                                           )

    val_generator = val_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                    target_size=(150, 150),
                                                    color_mode='rgb',
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    batch_size=8,
                                                    seed=123,
                                                    subset='validation'
                                                    )

    model = tf.keras.models.Sequential([
        # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        tf.keras.layers.Flatten(input_shape=(150, 150, 3)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    CALLBACK = myCallback()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )

    model.fit(train_generator,
              epochs=100,
              validation_data=val_generator,
              callbacks=[CALLBACK]
              )

    return model

# Save model as .h5 file
if __name__ == '__main__':
    model = solution_B3()
    model.save("model_B3.h5")
