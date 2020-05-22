"""
CNN model for 32x32x3 image classification
Uses smoothed inference defense when set to inference phase
"""
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from art.classifiers import KerasClassifier

from armory.data.utils import maybe_download_weights_from_s3


def preprocessing_fn(img):
    img = img.astype(np.float32) / 255.0
    return img


def _training_pass(x):
    x = Conv2D(
        filters=4,
        kernel_size=(5, 5),
        strides=1,
        activation="relu",
        input_shape=(32, 32, 3),
    )(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(
        filters=10,
        kernel_size=(5, 5),
        strides=1,
        activation="relu",
        input_shape=(23, 23, 4),
    )(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(100, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)
    return x


def _inference_pass(x):
    """
    Sample the base classifier's prediction under noisy corruptions of the input x.
    """
    x = Flatten()(x)
    x = Dense(10, activation="softmax")(x)
    return x


def make_cifar_model(**kwargs) -> tf.keras.Model:
    img_input = tf.keras.Input(shape=(32, 32, 3), name="img")

    # Conditional for handling training phase or inference phase
    output = tf.keras.backend.in_train_phase(
        _training_pass(img_input), _inference_pass(img_input)
    )

    model = tf.keras.Model(img_input, output)
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.003),
        metrics=["accuracy"],
    )
    return model


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = make_cifar_model(**model_kwargs)
    if weights_file:
        filepath = maybe_download_weights_from_s3(weights_file)
        model.load_weights(filepath)

    wrapped_model = KerasClassifier(model, clip_values=(0.0, 1.0), **wrapper_kwargs)
    return wrapped_model
