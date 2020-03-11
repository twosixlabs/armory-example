"""
Classifier evaluation within ARMORY
"""

import collections
from importlib import import_module
import json
import logging
import os
import sys

import numpy as np

from armory.utils.config_loading import load_dataset, load_model
from armory.eval import plot
from armory.paths import DockerPaths

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: Refactor (Issue #112)
def roc_targeted_epsilon(epsilons, min_epsilon=None, max_epsilon=None):
    if not len(epsilons):
        raise ValueError("epsilons cannot be empty")
    total = len(epsilons)
    epsilons = epsilons[epsilons != np.array(None)].astype(float)
    c = collections.Counter()
    c.update(epsilons)
    unique_epsilons, counts = zip(*sorted(list(c.items())))
    unique_epsilons = list(unique_epsilons)
    ccounts = np.cumsum(counts)
    targeted_attack_success = list((ccounts / total))

    if min_epsilon is not None and min_epsilon != unique_epsilons[0]:
        unique_epsilons.insert(0, min_epsilon)
        targeted_attack_success.insert(0, targeted_attack_success[0])
    if max_epsilon is not None and max_epsilon != unique_epsilons[-1]:
        unique_epsilons.append(max_epsilon)
        targeted_attack_success.append(
            targeted_attack_success[-1]
        )  # don't assume perfect attack success

    return (
        [float(x) for x in unique_epsilons],
        [float(x) for x in targeted_attack_success],
    )


def evaluate_classifier(config_path: str) -> None:
    """
    Evaluate a config file for classiifcation robustness against attack.
    """
    docker_paths = DockerPaths()

    with open(config_path, "r") as fp:
        config = json.load(fp)

    model_config = config["model"]
    classifier, preprocessing_fn = load_model(model_config)

    logger.info(f"Loading dataset {config['dataset']['name']}...")

    batch_size = config["dataset"]["batch_size"]

    train_data_generator = load_dataset(
        config["dataset"],
        split_type="train",
        epochs=config["adhoc"]["train_epochs"],
        preprocessing_fn=preprocessing_fn,
    )
    test_data_generator = load_dataset(
        config["dataset"],
        split_type="test",
        epochs=1,
        preprocessing_fn=preprocessing_fn,
    )

    logger.info(
        f"Fitting clean unpoisoned model of {model_config['module']}.{model_config['name']}..."
    )
    classifier.fit_generator(
        train_data_generator, nb_epochs=train_data_generator.total_iterations,
    )

    # Generate adversarial test examples
    attack_config = config["attack"]
    attack_module = import_module(attack_config["module"])
    attack_fn = getattr(attack_module, attack_config["name"])

    attack = attack_fn(classifier=classifier, **attack_config["kwargs"])
    norm = attack_config["budget"]["norm"][0]
    if norm == "L2":
        lp_norm = 2
    elif norm == "Linf":
        lp_norm = np.inf
    else:
        raise ValueError(
            f"Adversarial budget must have a norm of L2 or Linf. Found {norm} in config"
        )

    logger.info("Generating adversarial examples...")

    epsilons = []
    successful_attack_indices = []

    benign_misclassification_rate = 0
    targeted_attack_success_rate = 0
    clean_accuracy = 0
    cnt = 0

    # Sampling from the test set, so generate adversarial examples from the same batches
    # used to compute benign accuracy
    for _ in range(config["adhoc"]["num_attacked_pts"] // batch_size):
        x_test_sample, y_test_sample = test_data_generator.get_batch()

        y_target_sample = (y_test_sample + 1) % config["adhoc"]["num_classes"]

        x_test_adv = attack.generate(x=x_test_sample, y=y_target_sample)

        diff = (x_test_adv - x_test_sample).reshape(x_test_adv.shape[0], -1)
        epsilons.append(np.linalg.norm(diff, ord=lp_norm, axis=1))

        y_clean_pred = np.argmax(classifier.predict(x_test_sample), axis=1)
        y_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)

        # Evaluate the ART classifier on adversarial test examples and clean test examples
        successful_attack_indices_batch = (y_clean_pred != y_target_sample) & (
            y_adv_pred == y_target_sample
        )

        successful_attack_indices.append(successful_attack_indices_batch)

        benign_misclassification_rate += np.sum(
            y_clean_pred == y_target_sample
        ) / float(y_clean_pred.shape[0])

        targeted_attack_success_rate += np.sum(successful_attack_indices_batch) / float(
            y_clean_pred.shape[0]
        )
        clean_accuracy += np.sum(y_clean_pred == y_test_sample) / float(
            y_clean_pred.shape[0]
        )

        cnt += 1

    epsilons = np.concatenate(epsilons)
    successful_attack_indices = np.concatenate(successful_attack_indices)

    logger.info(
        f"Benign misclassification as targeted examples: {benign_misclassification_rate * 100 / cnt}%"
    )

    logger.info(f"Accuracy on benign test examples: {clean_accuracy * 100 / cnt}%")

    epsilons = epsilons.astype(object)
    epsilons[np.logical_not(successful_attack_indices)] = None

    unique_epsilons, targeted_attack_success = roc_targeted_epsilon(epsilons)
    results = {}

    results[norm] = {
        "epsilons": list(unique_epsilons),
        "metric": "Targeted attack success rate",
        "values": list(targeted_attack_success),
    }

    logger.info(
        f"Finished attacking on norm {norm}. Attack success: {targeted_attack_success_rate * 100/cnt}%"
    )

    logger.info("Saving json output...")
    filepath = os.path.join(
        docker_paths.output_dir, f"carlini_wagner_attack_{norm}_targeted_output.json"
    )
    with open(filepath, "w") as f:
        output_dict = {
            "config": config,
            "results": results,
        }
        json.dump(output_dict, f, sort_keys=True, indent=4)
    logger.info("Plotting results...")
    plot.classification(filepath)


if __name__ == "__main__":
    config_path = sys.argv[-1]
    evaluate_classifier(config_path)
