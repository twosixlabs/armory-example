"""
Example of loading two models pretrained weights in an ensemble
"""

import os
import tarfile

from art.classifiers import KerasClassifier
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Lambda

from armory.data.utils import maybe_download_weights_from_s3
from armory import paths

NUM_CLASSES = 45
SAVED_MODEL_DIR = paths.DockerPaths().saved_model_dir


def mean_std():
    resisc_mean = np.array(
        [0.36386173189316956, 0.38118692953271804, 0.33867067558870334,]
    )

    resisc_std = np.array([0.20350874, 0.18531173, 0.18472934])

    return resisc_mean, resisc_std

def make_ensemble_model(**model_kwargs) -> tf.keras.Model:
    input = tf.keras.Input(shape=(256, 256, 3))

    # Preprocessing layers
    img_scaled_to_255 = Lambda(lambda image: image * 255)(input)
    img_resized = Lambda(lambda image: tf.image.resize(image, (224, 224)))(
        img_scaled_to_255
    )
    img_scaled_to_1 = Lambda(lambda image: image / 255)(img_resized)
    mean, std = mean_std()
    img_standardized = Lambda(lambda image: (image - mean) / std)(img_scaled_to_1)

    model_names = ["model1", "model2"]
    models = []

    for model_name in model_names:
        model_notop = DenseNet121(include_top=False, weights=None, input_tensor=img_standardized)

        # Add new layers
        x = GlobalAveragePooling2D()(model_notop.output)
        predictions = Dense(NUM_CLASSES, activation="softmax")(x)

        # Create graph of new model
        new_model = Model(inputs=model_notop.input, outputs=predictions)

        # Load model weights
        new_model.load_weights(
            os.path.join(SAVED_MODEL_DIR, model_kwargs[f"{model_name}_weightsfile"])
        )
        for layer in new_model.layers:
            layer._name = layer.name + "_" + model_name

        models.append(new_model)

    # Average individual model results
    outputs = [model.output for model in models]
    ensemble_output = tf.keras.layers.Average()(outputs)
    ensemble_model = Model(input, ensemble_output, name="ensemble")

    # compile the ensemble
    ensemble_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return ensemble_model


def get_art_model(model_kwargs, wrapper_kwargs, weights_file):
    if weights_file:
        # Download tarball of all model weights
        filepath = maybe_download_weights_from_s3(weights_file)
        tar = tarfile.open(filepath)
        tar.extractall(path=SAVED_MODEL_DIR)
        tar.close()

        model = make_ensemble_model(**model_kwargs)
    else:
        raise NotImplementedError("This demo is for a pretrained ensemble")

    mean, std = mean_std()
    wrapped_model = KerasClassifier(
        model, clip_values=((0.0 - mean) / std, (1.0 - mean) / std), **wrapper_kwargs
    )
    return wrapped_model
