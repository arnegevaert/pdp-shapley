from itertools import combinations, chain
from typing import Tuple, Callable, Dict, Optional, Union, List
from pddshap.estimator import PDDEstimator, ConstantEstimator, LinearInterpolationEstimator, TreeEstimator, \
    ForestEstimator
from pddshap.coordinate_generator import CoordinateGenerator, EquidistantGridGenerator
import numpy as np
from tqdm import tqdm
from pddshap.coe import COECalculator
import pandas as pd


def _strict_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    # Note: this version only returns strict subsets
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)))


class ConstantPDDComponent:
    def __init__(self):
        self.estimator = None

    def fit(self, X: np.ndarray, model: Callable[[np.ndarray], np.ndarray]):
        self.estimator = ConstantEstimator(np.average(model(X), axis=0))

    def __call__(self, X: np.ndarray):
        return self.estimator(X)


class PDDComponent:
    def __init__(self, features: List[int], coordinate_generator, estimator_type: str) -> None:
        self.features = features
        self.estimator: Optional[PDDEstimator] = None
        self.std = 0
        self.fitted = False
        est_constructors = {
            "lin_interp": LinearInterpolationEstimator,
            "tree": TreeEstimator,
            "forest": ForestEstimator,
            "knn": None  # TODO
        }
        self.est_constructor = est_constructors[estimator_type]
        self.coordinate_generator = coordinate_generator

    def compute_partial_dependence(self, x: np.ndarray, X_bg: np.ndarray,
                                   model: Callable[[np.ndarray], np.ndarray]):
        # x: [len(self.features)]
        # X_bg: [n, num_features]
        X_copy = np.copy(X_bg)
        for i, feat in enumerate(self.features):
            X_copy[:, feat] = x[i]
        output = model(X_copy)
        if len(output.shape) == 1:
            output = output.reshape(output.shape[0], 1)
        return np.average(output, axis=0)

    def fit(self, X: np.ndarray, model: Callable[[np.ndarray], np.ndarray],
            subcomponents: Dict[Tuple[int], Union["PDDComponent", ConstantPDDComponent]]):
        # X: [n, num_features]
        # Define a grid of values
        coords = self.coordinate_generator.get_coords(X[:, self.features])

        # For each grid value, get partial dependence and subtract proper subset components
        partial_dependence = np.array([self.compute_partial_dependence(row, X, model) for row in coords])
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
            self.estimator = self.est_constructor()
            self.estimator.fit(coords, partial_dependence)
        self.fitted = True

    def __call__(self, X: pd.DataFrame):
        # X: [n, len(self.features)]
        if self.estimator is None:
            raise Exception("PDPComponent is not fitted yet")
        return self.estimator(X)


class PDDecomposition:
    def __init__(self, model: Callable[[np.ndarray], np.ndarray], coordinate_generator: CoordinateGenerator,
                 estimator_type: str) -> None:
        self.model = model
        self.components: Dict[Tuple, Union[ConstantPDDComponent, PDDComponent]] = {}
        if coordinate_generator is None:
            coordinate_generator = EquidistantGridGenerator(grid_res=10)
        self.coordinate_generator = coordinate_generator
        self.estimator_type = estimator_type

        self.bg_avg = None
        self.feature_names = None
        self.dtypes = None
        self.categories = None

    def extract_data_signature(self, X: pd.DataFrame):
        self.feature_names = list(X.columns)
        if len([col for col, dt in X.dtypes.items() if dt not in ["int8", "float32"]]) > 0:
            raise ValueError("Encode categorical values as int8 and numerical as float32")
        self.dtypes = X.dtypes
        self.categories = []
        for i, feat_name in enumerate(self.feature_names):
            if self.dtypes[i] == "float32":
                self.categories.append(0)
            else:
                self.categories.append(X[feat_name].nunique())

    def fit(self, X: pd.DataFrame, max_dim=None, eps=None) -> None:
        self.extract_data_signature(X)
        X_np = X.to_numpy()
        self.bg_avg = np.average(self.model(X_np), axis=0)

        coe_calculator = COECalculator(X_np, self.model)
        if max_dim is None:
            max_dim = len(self.feature_names)
        features = list(range(len(self.feature_names)))

        # self.average = np.average(self.model(X), axis=0)
        # Fit PDP components up to dimension max_dim
        for i in range(max_dim + 1):
            if i == 0:
                self.components[()] = ConstantPDDComponent()
                self.components[()].fit(X_np, self.model)
            else:
                # Get all subsets of given dimensionality
                subsets = list(combinations(features, i))
                # Create and fit a PDPComponent for each
                for subset in tqdm(subsets):
                    subset: List[int] = list(subset)
                    # subcomponents contains all PDPComponents for strict subsets of subset
                    subcomponents = {k: v for k, v in self.components.items() if all([feat in subset for feat in k])}
                    # Check if all subcomponents are present
                    if all([subcomponent in subcomponents for subcomponent in _strict_powerset(subset)]):
                        # If all subsets are significant, check if we need to fit this component
                        # TODO better design might be to have COECalculator compute all necessary components up front
                        # TODO for some given value of eps
                        coe = coe_calculator(subset) if eps is not None else 0
                        if eps is None or np.any(coe > eps):
                            # TODO pass relevant dtypes
                            self.components[tuple(subset)] = PDDComponent(subset, self.coordinate_generator,
                                                                          self.estimator_type)
                            self.components[tuple(subset)].fit(X_np, self.model, subcomponents)

    def evaluate(self, X: np.ndarray) -> Dict[Tuple[int], np.ndarray]:
        # X: [n, num_features]
        # Evaluate PDP decomposition at all rows in X
        # Returns each component function value separately
        result = {}
        for subset, component in self.components.items():
            result[subset] = component(X[:, subset])
        return result

    def __call__(self, X: pd.DataFrame):
        # TODO evaluate and aggregate (use PDDecomposition as surrogate model)
        pdp_values = self.evaluate(X.to_numpy())
        num_outputs = next(iter(pdp_values.items()))[1].shape[1]
        result = np.zeros((X.shape[0], num_outputs))
        for feature_subset, values in pdp_values.items():
            result += values
        return result

    # TODO this can be optimized, see linear_model.py
    def shapley_values(self, X: pd.DataFrame, project=False):
        result = []
        pdp_values = self.evaluate(X.to_numpy())
        # Infer the number of outputs from the decomposition output
        num_outputs = next(iter(pdp_values.items()))[1].shape[1]
        for col in X.columns:
            # [num_samples, num_outputs]
            # TODO am I double counting a bias here (ANOVA component for the empty set)?
            values_i = np.zeros((X.shape[0], num_outputs))
            col_idx = self.feature_names.index(col)
            for feature_subset, values in pdp_values.items():
                if col_idx in feature_subset:
                    values_i += values / len(feature_subset)
            result.append(values_i)
        # [num_samples, num_features, num_outputs]
        raw_values = np.stack(result, axis=1)
        if project:
            # Orthogonal projection of Shapley values onto hyperplane x_1 + ... + x_d = c
            # where c is the prediction difference
            pred_diff = (self.model(X) - self.bg_avg).reshape(-1, 1, raw_values.shape[-1])
            return raw_values - (np.sum(raw_values, axis=1, keepdims=True) - pred_diff) / X.shape[1]
        return raw_values
