import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from typing import Optional


class PDDEstimator:
    def __init__(self, categories):
        self.categories = categories

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ConstantEstimator(PDDEstimator):
    def __init__(self, output: np.ndarray):
        super().__init__(None)
        self.output = output

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        raise NotImplementedError

    def __call__(self, X):
        return np.tile(self.output, (X.shape[0], 1))


class TreeEstimator(PDDEstimator):
    def __init__(self, categories):
        super().__init__(categories)
        self.tree: Optional[DecisionTreeRegressor] = None

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        self.tree = DecisionTreeRegressor()
        self.tree.fit(coords, partial_dependence)

    def __call__(self, data):
        result = self.tree.predict(data)
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result


class ForestEstimator(PDDEstimator):
    def __init__(self, categories):
        super().__init__(categories)
        self.forest: Optional[RandomForestRegressor] = None

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        self.forest = RandomForestRegressor()
        self.forest.fit(coords, partial_dependence)

    def __call__(self, data):
        result = self.forest.predict(data)
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result


class KNNEstimator(PDDEstimator):
    def __init__(self, categories, k=3):
        super().__init__(categories)
        self.k = k
        self.knn: Optional[KNeighborsRegressor] = None

    def _preprocess(self, data):
        columns = []
        for i in range(data.shape[1]):
            if i not in self.categories:
                columns.append(data[:, i].reshape(-1, 1))
            else:
                ohe = OneHotEncoder(categories=[self.categories[i]], sparse=False)
                columns.append(ohe.fit_transform(data[:, i].reshape(-1, 1)))
        return np.concatenate(columns, axis=1)

    def fit(self, coords: np.ndarray, partial_dependence: np.ndarray):
        coords = self._preprocess(coords)
        self.knn = KNeighborsRegressor(n_neighbors=self.k)
        self.knn.fit(coords, partial_dependence)

    def __call__(self, data):
        result = self.knn.predict(self._preprocess(data))
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result
