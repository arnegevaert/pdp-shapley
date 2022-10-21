from itertools import combinations
from typing import Tuple, Callable, Dict, Union, List
from pddshap import ConstantPDDComponent, PDDComponent, CostOfExclusionEstimator, CoordinateGenerator, \
    EquidistantGridGenerator, FeatureSubset, DataSignature
import numpy as np
from tqdm import tqdm
import pandas as pd
from numpy import typing as npt


class PartialDependenceDecomposition:
    def __init__(self, model: Callable[[npt.NDArray], npt.NDArray], coordinate_generator: CoordinateGenerator,
                 estimator_type: str, est_kwargs=None) -> None:
        self.model = model
        self.components: Dict[FeatureSubset, Union[ConstantPDDComponent, PDDComponent]] = {}
        if coordinate_generator is None:
            coordinate_generator = EquidistantGridGenerator(grid_res=10)
        self.coordinate_generator = coordinate_generator
        self.estimator_type = estimator_type
        self.est_kwargs = est_kwargs if est_kwargs is not None else {}
        self.data_signature: DataSignature | None = None

        self.bg_avg = None
        self.num_outputs = None

    def fit(self, background_data: pd.DataFrame, max_cardinality=None, variance_explained=None) -> None:
        """
        Fit the partial dependence decomposition using a given background dataset.
        :param background_data:
        :param max_cardinality:
        :param variance_explained:
        :return:
        """
        self.data_signature = DataSignature(background_data)
        data_np = background_data.to_numpy()
        self.bg_avg = np.average(self.model(data_np), axis=0)

        significant_feature_sets = None
        if variance_explained is not None:
            coe_estimator = CostOfExclusionEstimator(data_np, self.model)
            significant_feature_sets = coe_estimator.get_significant_feature_sets(variance_explained, max_cardinality)
            max_cardinality = max(significant_feature_sets.keys())
        elif max_cardinality is None:
            max_cardinality = len(self.data_signature.feature_names)

        # Fit PDP components up to and including dimension max_dim
        for i in range(max_cardinality + 1):
            if i == 0:
                self.components[FeatureSubset()] = ConstantPDDComponent()
                self.components[FeatureSubset()].fit(data_np, self.model)
                self.num_outputs = self.components[FeatureSubset()].num_outputs
            else:
                # Get all subsets of given dimensionality
                feature_combs: List[Tuple]
                if significant_feature_sets is None:
                    feature_combs = list(combinations(range(self.data_signature.num_features), i))
                else:
                    feature_combs = significant_feature_sets[i]

                # Create and fit a PDPComponent for each
                for feature_comb in tqdm(feature_combs):
                    feature_subset = FeatureSubset(feature_comb)
                    # subcomponents contains all PDPComponents for strict subsets of feature_set
                    subcomponents = {k: v for k, v in self.components.items() if all([feat in feature_subset for feat in k])}
                    self.components[feature_subset] = PDDComponent(feature_subset, self.data_signature,
                                                                   self.coordinate_generator, self.estimator_type,
                                                                   self.est_kwargs)
                    self.components[feature_subset].fit(data_np, self.model, subcomponents)

    def __call__(self, data: pd.DataFrame | npt.NDArray):
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
            else:
                result += np.expand_dims(output_vector, axis=1)
        if project:
            # Orthogonal projection of Shapley values onto hyperplane x_1 + ... + x_d = c
            # where c is the prediction difference
            pred_diff = (self.model(data) - self.bg_avg).reshape(-1, 1, result.shape[-1])
            return result - (np.sum(result, axis=1, keepdims=True) - pred_diff) / data.shape[1]
        return result
