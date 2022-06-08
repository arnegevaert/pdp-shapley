from itertools import combinations, chain
from typing import Tuple, Callable, Dict, Optional, Union, List
from pddshap.estimator import PDDEstimator, ConstantEstimator, LinearInterpolationEstimator, TreeEstimator, \
    ForestEstimator
from pddshap.coordinate_generator import CoordinateGenerator, EquidistantGridGenerator
import numpy as np
from tqdm import tqdm
from pddshap.coe import COECalculator
import pandas as pd
from pddshap.preprocessor import Preprocessor


def _strict_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    # Note: this version only returns strict subsets
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)))


class ConstantPDDComponent:
    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor
        self.estimator = None

    def fit(self, X: pd.DataFrame, model: Callable[[pd.DataFrame], np.ndarray]):
        self.estimator = ConstantEstimator(np.average(model(self.preprocessor(X)), axis=0))

    def __call__(self, X: pd.DataFrame):
        return self.estimator(X)


class PDDComponent:
    def __init__(self, features: List[str], coordinate_generator, estimator_type: str, preprocessor: Preprocessor) -> None:
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
        self.preprocessor = preprocessor

    def compute_partial_dependence(self, x: np.ndarray, X_bg: pd.DataFrame,
                                   model: Callable[[pd.DataFrame], np.ndarray]):
        # x: [len(self.features)]
        # X: [n, num_features]
        X_copy = X_bg.copy(deep=True)
        X_copy[self.features] = x[self.features]
        output = model(self.preprocessor(X_copy))
        if len(output.shape) == 1:
            output = output.reshape(output.shape[0], 1)
        return np.average(output, axis=0)

    def fit(self, X: pd.DataFrame, model: Callable[[pd.DataFrame], np.ndarray],
            subcomponents: Dict[Tuple[int], Union["PDDComponent", ConstantPDDComponent]]):
        # X: [n, num_features]
        # Define a grid of values
        coords = self.coordinate_generator.get_coords(X[self.features]).drop_duplicates()

        # For each grid value, get partial dependence and subtract proper subset components
        partial_dependence = coords.apply(self.compute_partial_dependence, args=(X, model), axis=1, result_type="expand")
        for subset, subcomponent in subcomponents.items():
            if len(subset) == 0:
                partial_dependence -= subcomponent(coords)
            else:
                partial_dependence -= subcomponent(coords[list(subset)])

        # Fit a model on resulting values
        if ((partial_dependence.max() - partial_dependence.min()) < 1e-5).all():
            # If the partial dependence is constant, don't fit an interpolator
            self.estimator = ConstantEstimator(partial_dependence[0])
        else:
            # Otherwise, fit a model
            self.estimator = self.est_constructor()
            self.estimator.fit(self.preprocessor(coords), partial_dependence)
        self.fitted = True

    def __call__(self, X: pd.DataFrame):
        # X: [n, len(self.features)]
        if self.estimator is None:
            raise Exception("PDPComponent is not fitted yet")
        return self.estimator(self.preprocessor(X))


class PDDecomposition:
    def __init__(self, model: Callable[[pd.DataFrame], np.ndarray], coordinate_generator: CoordinateGenerator,
                 estimator_type: str, preprocessor: Preprocessor) -> None:
        self.model = model
        self.components: Dict[Tuple, Union[ConstantPDDComponent, PDDComponent]] = {}
        if coordinate_generator is None:
            coordinate_generator = EquidistantGridGenerator(grid_res=10)
        self.coordinate_generator = coordinate_generator
        self.estimator_type = estimator_type
        self.preprocessor = preprocessor

    def fit(self, X: pd.DataFrame, max_dim=None, eps=None) -> None:
        features = X.columns
        coe_calculator = COECalculator(X, self.model, self.preprocessor)
        if max_dim is None:
            max_dim = len(features)
        # self.average = np.average(self.model(X), axis=0)
        # Fit PDP components up to dimension max_dim
        for i in range(max_dim + 1):
            if i == 0:
                self.components[()] = ConstantPDDComponent(self.preprocessor)
                self.components[()].fit(X, self.model)
            else:
                print(f"Fitting {i}-dimensional components...")
                # Get all subsets of given dimensionality
                subsets = list(combinations(features, i))
                # Create and fit a PDPComponent for each
                for subset in tqdm(subsets):
                    subset: List[str] = list(subset)
                    # subcomponents contains all PDPComponents for strict subsets of subset
                    subcomponents = {k: v for k, v in self.components.items() if all([feat in subset for feat in k])}
                    # Check if all subcomponents are present
                    if all([subcomponent in subcomponents for subcomponent in _strict_powerset(subset)]):
                        # If all subsets are significant, check if we need to fit this component
                        # TODO better design might be to have COECalculator compute all necessary components up front
                        # TODO for some given value of eps
                        coe = coe_calculator(subset) if eps is not None else 0
                        if eps is None or np.any(coe > eps):
                            self.components[tuple(subset)] = PDDComponent(subset, self.coordinate_generator,
                                                                          self.estimator_type, self.preprocessor)
                            self.components[tuple(subset)].fit(X, self.model, subcomponents)

    def __call__(self, X: pd.DataFrame) -> Dict[Tuple[int], np.ndarray]:
        # X: [n, num_features]
        # Evaluate PDP decomposition at all rows in X
        # Returns each component function value separately
        result = {}
        for subset, component in self.components.items():
            if len(subset) > 0:
                result[subset] = component(X[list(subset)])
            else:
                result[subset] = component(X)
        return result
