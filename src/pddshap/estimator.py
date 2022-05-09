import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Optional


class PDDEstimator:
    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ConstantEstimator(PDDEstimator):
    def __init__(self, output: np.ndarray):
        super().__init__()
        self.output = output

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        raise NotImplementedError

    def __call__(self, X):
        return np.tile(self.output, (X.shape[0], 1))


class LinearInterpolationEstimator(PDDEstimator):
    def __init__(self):
        super().__init__()
        self.interpolator = None

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        if coords.shape[1] == 1:
            self.interpolator = interp1d(coords.flatten(), partial_dependence, fill_value="extrapolate", axis=0)
        else:
            # TODO extrapolate using nearest interpolator (create wrapper class)
            self.interpolator = LinearNDInterpolator(coords, partial_dependence, fill_value=0)

    def __call__(self, X):
        if X.shape[1] == 1:
            return self.interpolator(X.flatten())
        return self.interpolator(X)


class TreeEstimator(PDDEstimator):
    def __init__(self):
        super().__init__()
        self.tree: Optional[DecisionTreeRegressor] = None

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        self.tree = DecisionTreeRegressor()
        self.tree.fit(coords, partial_dependence)

    def __call__(self, X):
        result = self.tree.predict(X)
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result


class ForestEstimator(PDDEstimator):
    def __init__(self):
        super().__init__()
        self.forest: Optional[RandomForestRegressor] = None

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        self.forest = RandomForestRegressor()
        self.forest.fit(coords, partial_dependence)

    def __call__(self, X):
        result = self.forest.predict(X)
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result
