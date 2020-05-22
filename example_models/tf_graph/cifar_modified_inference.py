"""
CNN model for 32x32x3 image classification
Uses smoothed inference defense when set to inference phase
"""
import tarfile

import numpy as np
import tensorflow.compat.v1 as tf
from art.classifiers import TFClassifier

from armory import paths
from armory.data.utils import maybe_download_weights_from_s3


def preprocessing_fn(img):
    # Model will trained with inputs normalized from 0 to 1
    img = img.astype(np.float32) / 255.0
    return img


def _training_pass(x):
    x = tf.layers.conv2d(x, filters=4, kernel_size=(5, 5), activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.layers.conv2d(x, filters=10, kernel_size=(5, 5), activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 100, activation=tf.nn.relu)
    logits = tf.layers.dense(x, 10)
    return logits


def _inference_pass(x):
    """
    Sample the base classifier's prediction under noisy corruptions of the input x.
    """
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 100, activation=tf.nn.relu)
    logits = tf.layers.dense(x, 10)
    return logits


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    input_ph = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, shape=[None, 10])
    training_ph = tf.placeholder(tf.bool, shape=())

    # Conditional for handling training phase or inference phase
    output = tf.cond(
        training_ph,
        true_fn=lambda: _training_pass(input_ph),
        false_fn=lambda: _inference_pass(input_ph),
    )

    loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(logits=output, onehot_labels=labels_ph)
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.003)
    train_op = optimizer.minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if weights_file:
        # Load Model using preferred save/restore method
        filepath = maybe_download_weights_from_s3(weights_file)
        tar = tarfile.open(filepath)
        tar.extractall(path=paths.runtime_paths().saved_model_dir)
        tar.close()
        # Restore variables...

    wrapped_model = TFClassifier(
        clip_values=(0.0, 1.0),
        input_ph=input_ph,
        output=output,
        labels_ph=labels_ph,
        train=train_op,
        loss=loss,
        learning=training_ph,
        sess=sess,
        **wrapper_kwargs
    )

    return wrapped_model
