from itertools import combinations
from typing import Tuple, Callable, Dict, Union, List, Optional
from pddshap import ConstantPDDComponent, PDDComponent, FeatureSubsetSelector, CoordinateGenerator, \
    FeatureSubset, DataSignature
import numpy as np
from tqdm import tqdm
import pandas as pd
from numpy import typing as npt
from sklearn import cluster


class PartialDependenceDecomposition:
    def __init__(self, model: Callable[[npt.NDArray], npt.NDArray], coordinate_generator: CoordinateGenerator,
                 estimator_type: str, est_kwargs=None) -> None:
        self.model = model
        self.components: Dict[FeatureSubset, Union[ConstantPDDComponent, PDDComponent]] = {}
        self.coordinate_generator = coordinate_generator
        self.estimator_type = estimator_type
        self.est_kwargs = est_kwargs if est_kwargs is not None else {}
        self.data_signature: Optional[DataSignature] = None

        self.bg_avg = None
        self.num_outputs = None

    def fit(self, background_data: pd.DataFrame, train_data: pd.DataFrame, max_cardinality: int = None, variance_explained: float = 0.9,
            kmeans: int = None) -> None:
        """
        Fit the partial dependence decomposition using a given background dataset.
        :param background_data:
        :param max_cardinality:
        :param variance_explained:
        :param kmeans:
        :return:
        """
        self.data_signature = DataSignature(train_data)
        data_np = background_data.to_numpy()
        # Cluster the background distribution if necessary
        if kmeans is not None:
            data_np = cluster.KMeans(n_clusters=kmeans).fit(data_np).cluster_centers_
        self.bg_avg = np.average(self.model(data_np), axis=0)

        # Select subsets to be modeled
        subset_selector = FeatureSubsetSelector(train_data.to_numpy(), self.model)
        max_cardinality = max_cardinality if max_cardinality is not None else data_np.shape[1]
        significant_feature_sets = subset_selector.get_significant_feature_sets(variance_explained, max_cardinality)

        # Model the subsets in order of increasing cardinality
        for card in significant_feature_sets.keys():
            if card == 0:
                self.components[FeatureSubset()] = ConstantPDDComponent()
                self.components[FeatureSubset()].fit(data_np, self.model)
                self.num_outputs = self.components[FeatureSubset()].num_outputs
            else:
                for feature_set in significant_feature_sets[card]:
                    # All subcomponents are necessary to compute the values for this component
                    subcomponents = {k: v for k, v in self.components.items() if all([feat in feature_set for feat in k])}
                    self.components[feature_set] = PDDComponent(feature_set, self.data_signature,
                                                                   self.coordinate_generator, self.estimator_type,
                                                                   self.est_kwargs)
                    self.components[feature_set].fit(data_np, self.model, subcomponents)

    def __call__(self, data: pd.DataFrame | npt.NDArray):
        """
        Compute the output of the decomposition at the given coordinates
        :param data: The coordinates where we evaluate the model
        :return: The output of the decomposition at each coordinate.
        """
        if type(data) == pd.DataFrame:
            data = data.to_numpy()
        pdp_values = self.evaluate(data)
        result = None
        for feature_subset, values in pdp_values.items():
            if result is None:
                result = np.zeros((data.shape[0], values.shape[1]))
            result += values
        return result

    def evaluate(self, data: npt.NDArray) -> Dict[FeatureSubset, npt.NDArray]:
        """
        Evaluate PDP decomposition at all rows in data
        :param data: [num_rows, num_features]
        :return: Dictionary containing each component function value:
            {FeatureSubset, npt.NDArray[num_rows, num_outputs]}
        """
        return {subset: component(data) for subset, component in self.components.items()}

    def shapley_values(self, data: pd.DataFrame | npt.NDArray, project=False) -> npt.NDArray:
        """
        Compute Shapley values for each row in data.
        :param data: DataFrame or NDArray, shape: (num_rows, self.num_features)
        :param project: Boolean value indicating if the results should be orthogonally projected to the hyperplane
            satisfying the Efficiency axiom.
        :return: NDArray containing Shapley values for each row and each output.
            Shape: (num_rows, self.num_features, num_outputs)
        """
        if type(data) == pd.DataFrame:
            data = data.to_numpy()
        pdp_values = self.evaluate(data)

        result = np.zeros(shape=(data.shape[0], data.shape[1], self.num_outputs))
        for feature_subset, output_vector in pdp_values.items():
            if len(feature_subset) > 0:
                result[:, feature_subset.features, :] += np.expand_dims(output_vector, axis=1) / len(feature_subset)
        if project:
            # Orthogonal projection of Shapley values onto hyperplane x_1 + ... + x_d = c
            # where c is the prediction difference
            pred_diff = (self.model(data) - self.bg_avg).reshape(-1, 1, result.shape[-1])
            return result - (np.sum(result, axis=1, keepdims=True) - pred_diff) / data.shape[1]
        return result
