# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.83 and logs.get('val_accuracy') > 0.83):
            self.model.stop_training = True


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # Normalize your images here
    (feature_train, label_train), (feature_test, label_test) = fashion_mnist.load_data()

    # Normalize the data and add a channel dimension
    feature_train = feature_train / 255.0
    feature_test = feature_test / 255.0
    feature_train = tf.expand_dims(feature_train, axis=-1)  # Add channel dimension, shape becomes (28, 28, 1)
    feature_test = tf.expand_dims(feature_test, axis=-1)  # Same here for test data

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for Fashion MNIST
    ])

    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Create an instance of the callback
    callback = myCallback()

    # Train the model
    model.fit(feature_train,
              label_train,
              epochs=10,
              validation_data=(feature_test, label_test),
              callbacks=[callback])

    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
