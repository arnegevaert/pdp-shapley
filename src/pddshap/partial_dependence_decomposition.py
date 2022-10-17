from itertools import combinations
from typing import Tuple, Callable, Dict, Union, List
from pddshap.coordinate_generator import CoordinateGenerator, EquidistantGridGenerator
from pddshap import ConstantPDDComponent, PDDComponent, CostOfExclusionEstimator
import numpy as np
from tqdm import tqdm
import pandas as pd
from numpy import typing as npt


class PartialDependenceDecomposition:
    def __init__(self, model: Callable[[npt.NDArray], npt.NDArray], coordinate_generator: CoordinateGenerator,
                 estimator_type: str, est_kwargs=None) -> None:
        self.model = model
        self.components: Dict[Tuple, Union[ConstantPDDComponent, PDDComponent]] = {}
        if coordinate_generator is None:
            coordinate_generator = EquidistantGridGenerator(grid_res=10)
        self.coordinate_generator = coordinate_generator
        self.estimator_type = estimator_type
        self.est_kwargs = est_kwargs if est_kwargs is not None else {}

        self.bg_avg = None
        self.feature_names = None
        self.dtypes = None
        self.categories = None
        self.num_outputs = None

    def _extract_data_signature(self, dataframe: pd.DataFrame):
        """Extract information about data types (categorical vs numerical) from dataframe"""
        self.feature_names = list(dataframe.columns)
        if len([col for col, dt in dataframe.dtypes.items() if dt not in ["int8", "float32"]]) > 0:
            raise ValueError("Encode categorical values as int8 and numerical as float32")
        self.dtypes = dataframe.dtypes
        self.categories = []
        for i, feat_name in enumerate(self.feature_names):
            if self.dtypes[i] == "float32":
                self.categories.append([])
            else:
                self.categories.append(list(range(dataframe[feat_name].max() + 1)))

    def fit(self, background_data: pd.DataFrame, max_cardinality=None, variance_explained=None) -> None:
        """
        Fit the partial dependence decomposition using a given background dataset.
        :param background_data:
        :param max_cardinality:
        :param variance_explained:
        :return:
        """
        self._extract_data_signature(background_data)
        features = list(range(len(self.feature_names)))
        data_np = background_data.to_numpy()
        self.bg_avg = np.average(self.model(data_np), axis=0)

        significant_feature_sets = None
        if variance_explained is not None:
            coe_estimator = CostOfExclusionEstimator(data_np, self.model)
            significant_feature_sets = coe_estimator.get_significant_feature_sets(variance_explained, max_cardinality)
            max_cardinality = max(significant_feature_sets.keys())
        elif max_cardinality is None:
            max_cardinality = len(self.feature_names)

        # Fit PDP components up to and including dimension max_dim
        for i in range(max_cardinality + 1):
            if i == 0:
                self.components[()] = ConstantPDDComponent()
                self.components[()].fit(data_np, self.model)
                self.num_outputs = self.components[()].num_outputs
            else:
                # Get all subsets of given dimensionality
                feature_sets = list(combinations(features, i)) if significant_feature_sets is None else significant_feature_sets[i]
                # Create and fit a PDPComponent for each
                for feature_set in tqdm(feature_sets):
                    feature_set: List[int] = list(feature_set)
                    # subcomponents contains all PDPComponents for strict subsets of feature_set
                    subcomponents = {k: v for k, v in self.components.items() if all([feat in feature_set for feat in k])}
                    dtypes = [self.dtypes[i] for i in feature_set]
                    categories = [self.categories[i] for i in feature_set]
                    self.components[tuple(feature_set)] = PDDComponent(feature_set, self.coordinate_generator,
                                                                  self.estimator_type, dtypes, categories,
                                                                  self.est_kwargs)
                    self.components[tuple(feature_set)].fit(data_np, self.model, subcomponents)

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

    def evaluate(self, data: npt.NDArray) -> Dict[Tuple[int, ...], npt.NDArray]:
        """
        Evaluate PDP decomposition at all rows in data
        :param data: [num_rows, num_features]
        :return: Dictionary containing each component function value:
            {Tuple[int, ...], npt.NDArray[num_rows, num_outputs]}
        """
        result = {}
        for subset, component in self.components.items():
            result[subset] = component(data[:, subset])
        return result

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
        for feature_set, output_vector in pdp_values.items():
            if len(feature_set) > 0:
                result[:, feature_set, :] += np.expand_dims(output_vector, axis=1) / len(feature_set)
            else:
                result += np.expand_dims(output_vector, axis=1)
        if project:
            # Orthogonal projection of Shapley values onto hyperplane x_1 + ... + x_d = c
            # where c is the prediction difference
            pred_diff = (self.model(data) - self.bg_avg).reshape(-1, 1, result.shape[-1])
            return result - (np.sum(result, axis=1, keepdims=True) - pred_diff) / data.shape[1]
        return result