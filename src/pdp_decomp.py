import numpy as np
from itertools import combinations
from typing import Tuple, Callable, Dict
from scipy.interpolate import LinearNDInterpolator, interp1d
from tqdm import tqdm


class PDPComponent:
    def __init__(self, features: Tuple, eps: float) -> None:
        self.features = sorted(features)
        self.interpolator = None
        self.std = 0
        self.eps = eps
        self.sig = True
        self.fitted = False
    
    def compute_partial_dependence(self, x: np.ndarray, X_bg: np.ndarray, model: Callable[[np.ndarray], np.ndarray]):
        # x: [len(self.features)]
        # X: [n, num_features]
        X_copy = np.copy(X_bg)
        for i, feat in enumerate(self.features):
            X_copy[:, feat] = x[i]
        output = model(X_copy)
        if len(output.shape) == 1:
            output = output.reshape(output.shape[0], 1)
        return np.average(output, axis=0)

    def fit(self, X: np.ndarray, model: Callable[[np.ndarray], np.ndarray],
            subcomponents: Dict[Tuple[int], "PDPComponent"], grid_res=10):
        if len(self.features) == 0:
            avg_output = np.average(model(X), axis=0)
            self.interpolator = lambda inp: np.tile(avg_output, (inp.shape[0], 1))
        else:
            # X: [n, num_features]
            # Define a grid of values
            # Meshgrid creates coordinate matrices for each feature
            mg = np.meshgrid(*[np.linspace(np.min(X[:, feat]), np.max(X[:, feat]), grid_res) for feat in self.features])
            # Convert coordinate matrices to a single matrix containing a row for each grid point
            coords = np.vstack(list(map(np.ravel, mg))).transpose()

            # For each grid value, get partial dependence and subtract proper subset components
            pd = np.array([self.compute_partial_dependence(row, X, model) for row in coords])
            for subset, subcomponent in subcomponents.items():
                # features are given as indices in the full feature space
                # derive indices of pd that correspond to subcomponent features
                relative_indices = [self.features.index(feat) for feat in subset]

                # subtract subcomponent from partial dependence
                pd -= subcomponent(coords[:, relative_indices])

            # Check if this interaction is significant
            self.std = np.std(pd, axis=0)
            if self.features is not ():
                self.sig = (np.max(self.std) > self.eps)

            # Fit a model on resulting values 
            if np.max(pd) - np.min(pd) < 1e-5:
                # If the partial dependence is constant, don't fit an interpolator
                self.interpolator = lambda _: np.tile(pd[0], (X.shape[0], 1))
            else:
                if len(self.features) == 1:
                    self.interpolator = interp1d(coords.flatten(), pd, fill_value="extrapolate", axis=0)
                else:
                    self.interpolator = LinearNDInterpolator(coords, pd, fill_value=0)  # TODO extrapolate using nearest interpolator (create wrapper class)
        self.fitted = True

    def __call__(self, X: np.ndarray):
        # X: [n, len(self.features)]
        if self.interpolator is None:
            raise Exception("PDPComponent is not fitted yet")
        if len(self.features) == 1:
            return self.interpolator(X.flatten())
        return self.interpolator(X)


class PDPDecomposition:
    def __init__(self, model: Callable[[np.ndarray], np.ndarray]) -> None:
        self.model = model
        self.components: Dict[Tuple, PDPComponent] = {}
        self.average = 0

    def fit(self, X: np.ndarray, max_dim, eps) -> None:
        features = list(range(X.shape[1]))
        # self.average = np.average(self.model(X), axis=0)
        # Fit PDP components up to dimension max_dim
        for i in range(max_dim + 1):
            if i == 0:
                self.components[()] = PDPComponent((), eps)
                self.components[()].fit(X, self.model, {})
            else:
                print(f"Fitting {i}-dimensional components...")
                # Get all subsets of given dimensionality
                subsets = list(combinations(features, i))
                # Create and fit a PDPComponent for each
                for subset in tqdm(subsets):
                    subset: Tuple[int] = tuple(sorted(subset))
                    # subcomponents contains all PDPComponents for strict subsets of subset
                    subcomponents = {k: v for k, v in self.components.items() if all([feat in subset for feat in k])}
                    self.components[subset] = PDPComponent(subset, eps)
                    # Check if all subcomponents are significant
                    all_sig = all(subcomponents[k].sig for k in subcomponents)
                    if all_sig:
                        # If all subsets are significant, fit this component
                        self.components[subset].fit(X, self.model, subcomponents)
                    else:
                        # Otherwise, add component but mark as insignificant
                        # TODO we now add "dummy" components for insignificant interactions, this can be done more efficiently
                        self.components[subset].sig = False
    
    def __call__(self, X: np.ndarray) -> Dict[Tuple[int], np.ndarray]:
        # X: [n, num_features]
        # Evaluate PDP decomposition at all rows in X
        # Returns each component function value separately
        result = {}
        for subset, component in self.components.items():
            if component.fitted:
                result[subset] = component(X[:, subset])
        return result


class PDPShapleySampler:
    def __init__(self, model, X_background, num_outputs, max_dim=1, eps=0.05) -> None:
        self.model = model
        self.X_background = X_background

        self.pdp_decomp = PDPDecomposition(self.model)
        self.pdp_decomp.fit(self.X_background, max_dim, eps)
        self.num_outputs = num_outputs

    # TODO this can be optimized, see linear_model.py
    def estimate_shapley_values(self, X):
        result = []
        pdp_values = self.pdp_decomp(X)
        for i in range(X.shape[1]):
            # [num_samples, num_outputs]
            values_i = np.zeros((X.shape[0], self.num_outputs))
            for feature_subset, values in pdp_values.items():
                if i in feature_subset:
                    values_i += values / len(feature_subset)
            result.append(values_i)
        # [num_samples, num_features, num_outputs]
        return np.stack(result, axis=1)
