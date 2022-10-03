import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from typing import Optional


class PDDEstimator:
    def __init__(self, dtypes, categories):
        self.dtypes = dtypes,
        self.categories = categories

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ConstantEstimator(PDDEstimator):
    def __init__(self, output: np.ndarray):
        super().__init__(None, None)
        self.output = output

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        raise NotImplementedError

    def __call__(self, X):
        return np.tile(self.output, (X.shape[0], 1))


class TreeEstimator(PDDEstimator):
    def __init__(self, dtypes, categories):
        super().__init__(dtypes, categories)
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
    def __init__(self, dtypes, categories):
        super().__init__(dtypes, categories)
        self.forest: Optional[RandomForestRegressor] = None

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        self.forest = RandomForestRegressor()
        self.forest.fit(coords, partial_dependence)

    def __call__(self, X):
        result = self.forest.predict(X)
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result


class KNNEstimator(PDDEstimator):
    def __init__(self, dtypes, categories, k=3):
        super().__init__(dtypes, categories)
        self.k = k
        self.knn: Optional[KNeighborsRegressor] = None

    def _preprocess(self, X):
        columns = []
        for i in range(X.shape[1]):
            if len(self.categories[i]) == 0:
                columns.append(X[:, i].reshape(-1, 1))
            else:
                ohe = OneHotEncoder(categories=[self.categories[i]], sparse=False)
                columns.append(ohe.fit_transform(X[:, i].reshape(-1, 1)))
        return np.concatenate(columns, axis=1)

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        coords = self._preprocess(coords)
        self.knn = KNeighborsRegressor(n_neighbors=self.k)
        self.knn.fit(coords, partial_dependence)

    def __call__(self, X):
        result = self.knn.predict(self._preprocess(X))
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result
