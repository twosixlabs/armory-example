import numpy as np
from art.attacks.evasion import ProjectedGradientDescent


class CustomAttack(ProjectedGradientDescent):
    def __init__(self, estimator, **kwargs):
        modified_kwargs = kwargs.copy()
        modified_kwargs["targeted"] = True
        super().__init__(estimator, **modified_kwargs)

    def generate(self, x, y):

        for target in range(10):

            # Do not target correct class
            if target == y[0]:
                continue

            # Generate sample targeting `target` class
            y_target = np.zeros((1, 10), dtype=np.int64)
            y_target[0, target] = 1
            x_adv = super().generate(x, y_target)

            # Check - does this example fool the classifier?
            x_adv_pred = np.argmax(self.estimator.predict(x_adv))
            if x_adv_pred != y[0]:
                break

        return x_adv
