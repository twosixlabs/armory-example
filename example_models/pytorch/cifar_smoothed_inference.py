"""
CNN model for 32x32x3 image classification

Inference mode of training uses a non-differentiable smoothing.
"""
from math import ceil

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from art.classifiers import PyTorchClassifier

from armory.data.utils import maybe_download_weights_from_s3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_classes = 10
        self.conv1 = nn.Conv2d(3, 4, 5, 1)
        self.conv2 = nn.Conv2d(4, 10, 5, 1)
        self.fc1 = nn.Linear(250, 100)
        self.fc2 = nn.Linear(100, self.num_classes)

    def _training_pass(self, x):
        x = x.permute(0, 3, 1, 2)  # from NHWC to NCHW
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def _inference_pass(self, x, num, batch_size):
        """
        Sample the base classifier's prediction under noisy corruptions of the input x.
        """

        def _count_arr(arr, length: int):
            counts = np.zeros(length, dtype=int)
            for idx in arr:
                counts[idx] += 1
            return counts

        with torch.no_grad():
            counts = torch.zeros(self.num_classes, dtype=int)
            samples = ceil(num / batch_size)
            for _ in range(samples):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device=DEVICE) * 0.1
                predictions = self._training_pass(batch + noise).argmax(1)
                counts += _count_arr(predictions, self.num_classes)
        return torch.true_divide(counts, samples)

    def forward(self, x):
        # This flag is changed by ART's Classifier.set_learning_phase()
        if self.training:
            output = self._training_pass(x)
        else:
            output = self._inference_pass(x, 3, x.shape[0])

        return output


def make_cifar_model(**kwargs):
    return Net()


class PyTorchClassifierEstimatedInference(PyTorchClassifier):
    def loss_gradient(self, x, y, **kwargs):
        """
        This overrides the ART loss_gradient so that we use a differentiable
        approximation to compute the gradient during inference mode.

        Majority of the code is repurposed from PyTorchClassifier
        """

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Convert the inputs to Tensors
        inputs_t = torch.from_numpy(x_preprocessed).to(self._device)
        inputs_t.requires_grad = True

        # Convert the labels to Tensors
        labels_t = torch.from_numpy(y_preprocessed).to(self._device)

        # Compute the gradient
        if self._learning_phase:
            model_outputs = self._model(inputs_t)
            loss = self._loss(model_outputs[-1], labels_t)

        # Estimate the inference time gradient since it is a non-differentiable defense
        else:
            model_outputs = self._model._model._training_pass(inputs_t)
            loss = self._loss(model_outputs, labels_t)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = inputs_t.grad.cpu().numpy().copy()
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x.shape

        return grads


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = make_cifar_model(**model_kwargs)
    model.to(DEVICE)

    if weights_file:
        filepath = maybe_download_weights_from_s3(weights_file)
        checkpoint = torch.load(filepath, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchClassifierEstimatedInference(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
