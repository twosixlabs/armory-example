"""
General audio classification scenario using spectrograms

This way of approaching the scenario requires augmenting the label set of the data.
The baseline audio classification scenario does not support this so a new scenario
was created.

Scenario contributed by: MITRE Corporation
"""

import logging
from typing import Optional

import numpy as np
from tqdm import tqdm

from armory.utils.config_loading import (
    load_dataset,
    load_model,
    load_attack,
    load_defense_internal,
)
from armory.utils import metrics
from armory.scenarios.base import Scenario

logger = logging.getLogger(__name__)


class AudioSpectrogramClassificationTask(Scenario):
    def _evaluate(
        self, config: dict, num_eval_batches: Optional[int], skip_benign: Optional[bool]
    ) -> dict:
        """
        Evaluate a config file for classification robustness against attack.
        """
        model_config = config["model"]
        classifier, _ = load_model(model_config)

        defense_config = config.get("defense") or {}
        defense_type = defense_config.get("type")

        if defense_type in ["Preprocessor", "Postprocessor"]:
            logger.info(f"Applying internal {defense_type} defense to classifier")
            classifier = load_defense_internal(config["defense"], classifier)

        task_metric = metrics.categorical_accuracy

        if config["dataset"]["batch_size"] != 1:
            raise NotImplementedError("Currently only supports batch size of 1")

        # Train ART classifier
        if not model_config["weights_file"]:
            raise NotImplementedError("Gradients not available for training.")
            classifier.set_learning_phase(True)
            logger.info(
                f"Fitting model {model_config['module']}.{model_config['name']}..."
            )
            fit_kwargs = model_config["fit_kwargs"]
            train_data_generator = load_dataset(
                config["dataset"], epochs=fit_kwargs["nb_epochs"], split_type="train",
            )

            for cnt, (x, y) in tqdm(enumerate(train_data_generator)):
                classifier.fit(
                    x,
                    y,
                    batch_size=config["dataset"]["batch_size"],
                    nb_epochs=1,
                    verbose=True,
                )

                if (cnt + 1) % train_data_generator.batches_per_epoch == 0:
                    # evaluate on validation examples
                    val_data_generator = load_dataset(
                        config["dataset"], epochs=1, split_type="validation",
                    )

                    cnt = 0
                    validation_accuracies = []
                    for x_val, y_val in tqdm(val_data_generator):
                        y_pred = np.mean(
                            classifier.predict(x_val, batch_size=1),
                            axis=0,
                            keepdims=True,
                        )
                        validation_accuracies.extend(task_metric(y_val, y_pred))
                        cnt += len(y_val)
                    validation_accuracy = sum(validation_accuracies) / cnt
                    logger.info("Validation accuracy: {}".format(validation_accuracy))

        classifier.set_learning_phase(False)
        # Evaluate ART classifier on test examples
        if skip_benign:
            logger.info("Skipping benign classification...")
        else:
            logger.info(f"Loading testing dataset {config['dataset']['name']}...")
            test_data_generator = load_dataset(
                config["dataset"],
                epochs=1,
                split_type="test",
                num_batches=num_eval_batches,
            )
            logger.info("Running inference on benign test examples...")

            cnt = 0
            benign_accuracies = []
            for x, y in tqdm(test_data_generator, desc="Benign"):
                y_pred = np.mean(
                    classifier.predict(x, batch_size=1), axis=0, keepdims=True
                )
                benign_accuracies.extend(task_metric(y, y_pred))
                cnt += len(y)

            benign_accuracy = sum(benign_accuracies) / cnt
            logger.info(f"Accuracy on benign test examples: {benign_accuracy:.2%}")

        # Evaluate the ART classifier on adversarial test examples
        logger.info("Generating / testing adversarial examples...")
        attack = load_attack(config["attack"], classifier)

        test_data_generator = load_dataset(
            config["dataset"], epochs=1, split_type="test", num_batches=num_eval_batches
        )

        cnt = 0
        adversarial_accuracies = []
        for x, y in tqdm(test_data_generator, desc="Attack"):
            x_adv = attack.generate(x=x)
            y_pred = np.mean(
                classifier.predict(x_adv, batch_size=1), axis=0, keepdims=True
            )
            adversarial_accuracies.extend(task_metric(y, y_pred))
            cnt += len(y)
        adversarial_accuracy = sum(adversarial_accuracies) / cnt
        logger.info(
            f"Accuracy on adversarial test examples: {adversarial_accuracy:.2%}"
        )
        results = {"mean_adversarial_accuracy": adversarial_accuracy}
        if not skip_benign:
            results["mean_benign_accuracy"] = benign_accuracy
        return results
