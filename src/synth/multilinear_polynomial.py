import numpy as np


class MultilinearPolynomial:
    def __init__(self, num_features: int):
        self.num_features = num_features

    def __call__(self, data: np.ndarray):
        pass

    def shapley_values(self, data: np.ndarray):
        """
        Computes closed form Shapley values for inputs given by X (shape: [-1, self.num_features]).
        This assumes that the input features are centered around 0 and independently distributed.
        """
        pass