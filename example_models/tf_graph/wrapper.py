"""
Tensorflow1 model wrapper for Image classification on CIFAR10

Contributed by: Tianji Cong, University of Michigan
"""
import os

import numpy as np
import tensorflow as tf
from armory import paths
from art.classifiers import TFClassifier

from tf_graph.model import make_madry_model


def preprocessing_fn(img):
    img = img.astype(np.float32)

    return img


def get_madry_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = make_madry_model(**model_kwargs)
    input_ph = model.x_input
    labels_ph = model.y_input
    training_ph = tf.placeholder(tf.bool, shape=())

    # Restore the checkpoint
    saver = tf.train.Saver()
    tf_sess = tf.Session()

    saved_model_dir = paths.DockerPaths().saved_model_dir
    filepath = os.path.join(saved_model_dir, weights_file)
    model_file = tf.train.latest_checkpoint(filepath)
    saver.restore(tf_sess, model_file)

    wrapped_model = TFClassifier(
        input_ph=input_ph,
        output=model.pre_softmax,
        labels_ph=labels_ph,
        loss=model.xent,
        learning=training_ph,
        sess=tf_sess,
        clip_values=(0, 255),
        **wrapper_kwargs
    )

    return wrapped_model
