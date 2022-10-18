from typing import Tuple, Callable, Dict, Optional, Union, List
from pddshap import PDDEstimator, ConstantEstimator, TreeEstimator, \
    ForestEstimator, KNNEstimator
import numpy as np
from numpy import typing as npt
import pandas as pd


class PDDComponent:
    def __init__(self, features: List[int], coordinate_generator, estimator_type: str, dtypes, categories, est_kwargs) -> None:
        self.features = features
        self.estimator: Optional[PDDEstimator] = None
        self.std = 0
        self.fitted = False
        est_constructors = {
            "tree": TreeEstimator,
            "forest": ForestEstimator,
            "knn": KNNEstimator
        }
        self.est_constructor = est_constructors[estimator_type]
        self.coordinate_generator = coordinate_generator
        self.dtypes = dtypes
        self.categories = categories
        self.est_kwargs = est_kwargs
        self.num_outputs = None

    def _compute_partial_dependence(self, x: npt.NDArray, background_distribution: npt.NDArray,
                                    model: Callable[[npt.NDArray], npt.NDArray]):
        """
        Compute partial dependence of this component at a given point.
        :param x: The point at which we compute the partial dependence. Shape: (num_features,)
        :param background_distribution: The background distribution to estimate the integral.
            Shape: (-1, num_features)
        :param model: The model of which we want to compute the partial dependence
        :return: Partial dependence at x. Shape: (num_outputs,)
        """
        bgd_copy = np.copy(background_distribution)
        for i, feat in enumerate(self.features):
            bgd_copy[:, feat] = x[i]
        output = model(bgd_copy)  # (bgd_copy.shape[0], num_outputs)
        if len(output.shape) == 1:
            # If there is only 1 output, the dimension must be added
            output = output.reshape(output.shape[0], 1)
        return np.average(output, axis=0)

    def fit(self, data: np.ndarray, model: Callable[[np.ndarray], np.ndarray],
            subcomponents: Dict[Tuple[int], Union["PDDComponent", "ConstantPDDComponent"]]):
        """

        :param data: [n, num_features]
        :param model:
        :param subcomponents:
        :return:
        """
        # Define a grid of values
        coords = self.coordinate_generator.get_coords(data[:, self.features])

        # For each grid value, get partial dependence and subtract proper subset components
        # Shape: (num_rows, num_outputs)
        # TODO can we use jit compilation to make this faster?
        partial_dependence = np.array([self._compute_partial_dependence(row, data, model) for row in coords])
        self.num_outputs = partial_dependence.shape[1]
        for subset, subcomponent in subcomponents.items():
            # Features are given as indices in the full feature space
            # Derive indices of partial_dependence that correspond to subcomponent features
            relative_indices = [self.features.index(feat) for feat in subset]
            # Subtract subcomponent from partial dependence
            partial_dependence -= subcomponent(coords[:, relative_indices])

        # Fit a model on resulting values
        if np.max(partial_dependence) - np.min(partial_dependence) < 1e-5:
            # If the partial dependence is constant, don't fit an interpolator
            self.estimator = ConstantEstimator(partial_dependence[0])
        else:
            # Otherwise, fit a model
            self.estimator = self.est_constructor(self.dtypes, self.categories, **self.est_kwargs)
            self.estimator.fit(coords, partial_dependence)
        self.fitted = True

    def __call__(self, data: pd.DataFrame) -> npt.NDArray:
        """

        :param data:
        :return: Shape: (data.shape[0], self.num_outputs)
        """
        # X: [n, len(self.features)]
        if self.estimator is None:
            raise Exception("PDPComponent is not fitted yet")
        return self.estimator(data)


class ConstantPDDComponent:
    def __init__(self):
        self.estimator = None
        self.num_outputs = None

    def fit(self, data: np.ndarray, model: Callable[[npt.NDArray], npt.NDArray]):
        avg_output = np.average(model(data), axis=0)
        self.estimator = ConstantEstimator(avg_output)
        self.num_outputs = avg_output.shape[0] if type(avg_output) == npt.NDArray else 1

    def __call__(self, data: np.ndarray):
        return self.estimator(data)

