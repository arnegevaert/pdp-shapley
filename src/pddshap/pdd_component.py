from typing import Tuple, Callable, Dict, Optional, Union
from pddshap import PDDEstimator, ConstantEstimator, TreeEstimator, \
    ForestEstimator, KNNEstimator, FeatureSubset, DataSignature
import numpy as np
from numpy import typing as npt
import pandas as pd


class PDDComponent:
    def __init__(self, feature_subset: FeatureSubset, data_signature: DataSignature,
                 coordinate_generator, estimator_type: str, est_kwargs) -> None:
        self.data_signature = data_signature
        self.feature_subset = feature_subset
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
        output = model(self.feature_subset.project(background_distribution, x))  # (num_rows, num_outputs)
        if len(output.shape) == 1:
            # If there is only 1 output, the dimension must be added
            output = np.expand_dims(output, axis=-1)
        return np.average(output, axis=0)

    def fit(self, data: np.ndarray, model: Callable[[np.ndarray], np.ndarray],
            subcomponents: Dict[FeatureSubset, Union["PDDComponent", "ConstantPDDComponent"]]):
        """

        :param data: [n, num_features]
        :param model:
        :param subcomponents:
        :return:
        """
        # Define a grid of values
        # TODO this returns coordinates in the original feature space.
        #       Not a problem for now, but should be replaced with a more general "Quasi Monte Carlo Integration" class
        coords = self.coordinate_generator.get_coords(data)

        # For each grid value, get partial dependence and subtract proper subset components
        # Shape: (num_rows, num_outputs)
        # TODO can we use jit compilation to make this faster?
        partial_dependence = np.array([self._compute_partial_dependence(row, data, model) for row in coords])
        self.num_outputs = partial_dependence.shape[1]
        for subset, subcomponent in subcomponents.items():
            # Subtract subcomponent from partial dependence
            partial_dependence -= subcomponent(coords)

        # Fit a model on resulting values
        if np.max(partial_dependence) - np.min(partial_dependence) < 1e-5:
            # If the partial dependence is constant, don't fit an estimator
            self.estimator = ConstantEstimator(partial_dependence[0])
        else:
            # Otherwise, fit a model
            categories = self.data_signature.get_categories(self.feature_subset)
            self.estimator = self.est_constructor(categories, **self.est_kwargs)
            self.estimator.fit(self.feature_subset.get_columns(coords), partial_dependence)
        self.fitted = True

    def __call__(self, data: pd.DataFrame | npt.NDArray) -> npt.NDArray:
        """
        Computes this component's output on the given data.
        Note: all columns must be passed in the data. The method extracts the relevant columns.
        :param data: Shape: (num_rows, len(self.data_signature.feature_names)
        :return: Shape: (data.shape[0], self.num_outputs)
        """
        if self.estimator is None:
            raise Exception("PDPComponent is not fitted yet")
        return self.estimator(self.feature_subset.get_columns(data))


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