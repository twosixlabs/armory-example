"""
Preprocessing and simple model architecture for German Traffic Sign Recognition Benchmark
"""
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from art.classifiers import KerasClassifier


def preprocessing_fn(img):
    img_size = 48
    img_out = []
    for im in img:
        im = Image.fromarray(im)
        im = np.array(im.resize((img_size, img_size)))
        img_out.append(im)
    return np.array(img_out, dtype=np.float32)


def make_model(**kwargs) -> tf.keras.Model:
    model = Sequential()
    model.add(
        Conv2D(
            filters=4,
            kernel_size=(5, 5),
            strides=1,
            activation="relu",
            input_shape=(48, 48, 3),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            filters=10,
            kernel_size=(5, 5),
            strides=1,
            activation="relu",
            input_shape=(22, 22, 4),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(43, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.003),
        metrics=["accuracy"],
    )
    return model


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = make_model(**model_kwargs)
    wrapped_model = KerasClassifier(model, clip_values=(0.0, 1.0), **wrapper_kwargs)
    return wrapped_model
