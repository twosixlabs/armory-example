"""
CNN model for 241x100x1 audio spectrogram classification

Model contributed by: MITRE Corporation
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from art.classifiers import KerasClassifier

from armory.data.utils import maybe_download_weights_from_s3


def make_model(**kwargs) -> tf.keras.Model:
    model = Sequential()
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=1,
            activation="relu",
            input_shape=(241, 100, 1),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=1, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(40, activation="softmax"))

    model.compile(
        loss=keras.losses.sparse_categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=0.0002),
        metrics=["accuracy"],
    )
    return model


def get_art_model(model_kwargs, wrapper_kwargs, weights_file):
    model = make_model(**model_kwargs)
    if weights_file:
        filepath = maybe_download_weights_from_s3(weights_file)
        model.load_weights(filepath)

    wrapped_model = KerasClassifier(model, clip_values=(-1.0, 1.0), **wrapper_kwargs)
    return wrapped_model
