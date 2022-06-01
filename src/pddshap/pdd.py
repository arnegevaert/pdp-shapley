from itertools import combinations, chain
from typing import Tuple, Callable, Dict, Optional
from pddshap.estimator import PDDEstimator, ConstantEstimator, LinearInterpolationEstimator, TreeEstimator, ForestEstimator
from pddshap.coordinate_generator import CoordinateGenerator, EquidistantGridGenerator
import numpy as np
from tqdm import tqdm
from pddshap.coe import cost_of_exclusion


def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    # Note: this version only returns strict subsets
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)))


class PDDComponent:
    def __init__(self, features: Tuple, coordinate_generator, estimator_type) -> None:
        self.features = sorted(features)
        self.estimator: Optional[PDDEstimator] = None
        self.std = 0
        self.fitted = False
        est_constructors = {
            "lin_interp": LinearInterpolationEstimator,
            "tree": TreeEstimator,
            "forest": ForestEstimator
        }
        self.est_constructor = est_constructors[estimator_type]
        self.coordinate_generator = coordinate_generator

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
            subcomponents: Dict[Tuple[int], "PDDComponent"]):
        if len(self.features) == 0:
            self.estimator = ConstantEstimator(np.average(model(X), axis=0))
        else:
            # X: [n, num_features]
            # Define a grid of values
            coords = self.coordinate_generator.get_coords(X[:, self.features])

            # For each grid value, get partial dependence and subtract proper subset components
            pd = np.array([self.compute_partial_dependence(row, X, model) for row in coords])
            for subset, subcomponent in subcomponents.items():
                # features are given as indices in the full feature space
                # derive indices of pd that correspond to subcomponent features
                relative_indices = [self.features.index(feat) for feat in subset]

                # subtract subcomponent from partial dependence
                pd -= subcomponent(coords[:, relative_indices])

            # Fit a model on resulting values
            if np.max(pd) - np.min(pd) < 1e-5:
                # If the partial dependence is constant, don't fit an interpolator
                self.estimator = ConstantEstimator(pd[0])
            else:
                # Otherwise, fit a linear interpolator
                self.estimator = self.est_constructor()
                self.estimator.fit(coords, pd)
        self.fitted = True

    def __call__(self, X: np.ndarray):
        # X: [n, len(self.features)]
        if self.estimator is None:
            raise Exception("PDPComponent is not fitted yet")
        return self.estimator(X)


class PDDecomposition:
    def __init__(self, model: Callable[[np.ndarray], np.ndarray], coordinate_generator: CoordinateGenerator, estimator_type: str) -> None:
        self.model = model
        self.components: Dict[Tuple, PDDComponent] = {}
        self.average = 0
        if coordinate_generator is None:
            coordinate_generator = EquidistantGridGenerator(grid_res=10)
        self.coordinate_generator = coordinate_generator
        self.estimator_type = estimator_type

    def fit(self, X: np.ndarray, max_dim=None, eps=None) -> None:
        features = list(range(X.shape[1]))
        if max_dim is None:
            max_dim = len(features)
        # self.average = np.average(self.model(X), axis=0)
        # Fit PDP components up to dimension max_dim
        for i in range(max_dim + 1):
            if i == 0:
                self.components[()] = PDDComponent((), self.coordinate_generator, self.estimator_type)
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
                    # Check if all subcomponents are present
                    if all([subcomponent in subcomponents for subcomponent in _powerset(subset)]):
                        # If all subsets are significant, check if we need to fit this component
                        coe = cost_of_exclusion(X, subset, self.model) if eps is not None else 0
                        print(subset, coe)
                        if eps is None or coe > eps:
                            self.components[subset] = PDDComponent(subset, self.coordinate_generator,
                                                                   self.estimator_type)
                            self.components[subset].fit(X, self.model, subcomponents)

    def __call__(self, X: np.ndarray) -> Dict[Tuple[int], np.ndarray]:
        # X: [n, num_features]
        # Evaluate PDP decomposition at all rows in X
        # Returns each component function value separately
        result = {}
        for subset, component in self.components.items():
            if component.fitted:
                result[subset] = component(X[:, subset])
        return result


