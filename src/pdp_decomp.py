import numpy as np
from itertools import combinations
from typing import Tuple, Callable, Dict
from scipy.interpolate import LinearNDInterpolator, interp1d
from tqdm import tqdm


class PDPComponent:
    def __init__(self, features: Tuple[int]) -> None:
        self.features = sorted(features)
        self.interpolator = None
    
    def compute_partial_dependence(self, x: np.ndarray, X_bg: np.ndarray, model: Callable[[np.ndarray], np.ndarray]):
        # x: [len(self.features)]
        # X: [n, num_features]
        X_copy = np.copy(X_bg)
        for i, feat in enumerate(self.features):
            X_copy[:, feat] = x[i]
        return np.average(model(X_copy), axis=0)

    def fit(self, X: np.ndarray, model: Callable[[np.ndarray], np.ndarray],
            subcomponents: Dict[Tuple[int], "PDPComponent"], grid_res=10):
        if len(self.features) == 0:
            avg_output = np.average(model(X))
            self.interpolator = lambda _: avg_output
        else:
            # X: [n, num_features]
            # Define a grid of values
            # Meshgrid creates coordinate matrices for each feature
            mg = np.meshgrid(*[np.linspace(np.min(X[:, feat]), np.max(X[:, feat]), grid_res) for feat in self.features])
            # Convert coordinate matrices to a single matrix containing a row for each grid point
            coords = np.vstack(list(map(np.ravel, mg))).transpose()

            # For each grid value, get partial dependence and subtract proper subset components
            pd = np.array([self.compute_partial_dependence(row, X, model) for row in coords]).flatten()
            for subset, subcomponent in subcomponents.items():
                # features are given as indices in the full feature space
                # derive indices of pd that correspond to subcomponent features
                relative_indices = [self.features.index(feat) for feat in subset]

                # subtract subcomponent from partial dependence
                pd -= subcomponent(coords[:, relative_indices])

            # Fit a model on resulting values 
            if np.max(pd) - np.min(pd) < 1e-5:
                # If the partial dependence is constant, don't fit an interpolator
                self.interpolator = lambda _: pd[0]
            else:
                if len(self.features) == 1:
                    self.interpolator = interp1d(coords.flatten(), pd, fill_value="extrapolate")
                else:
                    self.interpolator = LinearNDInterpolator(coords, pd, fill_value=0)  # TODO extrapolate using nearest interpolator (create wrapper class)

    def __call__(self, X: np.ndarray):
        # X: [n, len(self.features)]
        if self.interpolator is None:
            raise("PDPComponent is not fitted yet")
        return self.interpolator(X).flatten()


class PDPDecomposition:
    def __init__(self, model: Callable[[np.ndarray], np.ndarray]) -> None:
        self.model = model
        self.components: Dict[Tuple[int], PDPComponent] = {}
        self.average = 0

    def fit(self, X: np.ndarray, max_dim) -> None:
        features = list(range(X.shape[1]))
        # self.average = np.average(self.model(X), axis=0)
        # Fit PDP components up to dimension max_dim
        for i in range(max_dim + 1):
            if i == 0:
                self.components[()] = PDPComponent(())
                self.components[()].fit(X, self.model, {})
            else:
                print(f"Fitting {i}-dimensional components...")
                # Get all subsets of given dimensionality
                subsets = list(combinations(features, i))
                # Create and fit a PDPComponent for each
                for subset in tqdm(subsets):
                    subset = tuple(sorted(subset))
                    # subcomponents contains all PDPComponents for strict subsets of subset
                    subcomponents = {k: v for k, v in self.components.items() if all([feat in subset for feat in k])}
                    self.components[subset] = PDPComponent(subset)
                    self.components[subset].fit(X, self.model, subcomponents)
    
    def __call__(self, X: np.ndarray) -> Dict[Tuple[int], np.ndarray]:
        # X: [n, num_features]
        # Evaluate PDP decomposition at all rows in X
        # Returns each component function value separately
        result = {}
        for subset, component in self.components.items():
            result[subset] = component(X[:, subset])# - self.average
        return result


class PDPShapleySampler:
    def __init__(self, model, X_background, max_dim=1) -> None:
        self.model = model
        self.X_background = X_background

        self.pdp_decomp = PDPDecomposition(self.model)
        self.pdp_decomp.fit(self.X_background, max_dim)

    def estimate_shapley_values(self, X):
        result = []
        pdp_values = self.pdp_decomp(X)
        for i in range(X.shape[1]):
            values_i = np.zeros(X.shape[0])
            for feature_subset, values in pdp_values.items():
                if i in feature_subset:
                    values_i += values / len(feature_subset)
            result.append(values_i.reshape(-1, 1))
        return np.hstack(result)