import numpy as np
from itertools import combinations
from typing import List, Callable, Dict, Set


class PDPComponent:
    def __init__(self, features: Set[int]) -> None:
        self.features = features

    def fit(self, X: np.ndarray, subcomponents: Dict[Set[int], "PDPComponent"]):
        # Define a grid of values

        # For each grid value, get partial dependence

        # For each grid value, subtract all proper subset components

        # Fit a model on resulting values 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html#scipy.interpolate.LinearNDInterpolator
        pass

    def __call__(self, x: np.ndarray):
        pass


class PDPDecomposition:
    def __init__(self, model: Callable[[np.ndarray], np.ndarray]) -> None:
        self.model = model
        self.components: Dict[Set[int], PDPComponent] = {}

    def fit(self, X: np.ndarray, max_dim) -> None:
        features = list(range(X.shape[1]))
        # Fit PDP components up to dimension max_dim
        for i in range(max_dim):
            dim = i + 1
            # Get all subsets of given dimensionality
            subsets = combinations(features, dim)
            # Create and fit a PDPComponent for each
            for subset in subsets:
                subset = set(subset)
                # subcomponents contains all PDPComponents for strict subsets of subset
                subcomponents = {k: v for k, v in self.components if all([feat in subset for feat in k])}
                self.components[subset] = PDPComponent(subset)
                self.components[subset].fit(X, subcomponents)
    
    def __call__(self, x: np.ndarray) -> Dict[Set[int], np.ndarray]:
        # Evaluate PDP decomposition at x
        # Returns each component function value separately
        result = {}
        for subset, component in self.components:
            result[subset] = component(x)
        return result


class PDPShapleySampler:
    def __init__(self, model, X_background, max_dim=1) -> None:
        self.model = model
        self.X_background = X_background

        self.pdp_decomp = PDPDecomposition(self.model)
        self.pdp_decomp.fit(self.X_background, max_dim)

    def estimate_shapley_values(self, X):
        result = []
        for row in X:
            # TODO this code does not yet account for multivariate outputs
            pdp_values = self.pdp_decomp(row)
            row_shapley_values = []
            for i in range(X.shape[1]):
                value_i = 0
                for feature_subset, value in pdp_values:
                    if i in feature_subset:
                        value_i += value / len(feature_subset)
                row_shapley_values.append(value_i)
            result.append(row_shapley_values)