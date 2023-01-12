import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from pddshapley.signature import FeatureSubset
from typing import Optional, Dict, List
from numpy import typing as npt


Categories = Dict[int, List[int]]

# TODO estimators should get the full data frame and select their columns


class PDDEstimator:
    def __init__(self, categories: Optional[Categories], feature_subset: Optional[FeatureSubset]):
        self.feature_subset = feature_subset
        self.categories = categories

    def fit(self, coords: npt.NDArray, partial_dependence: npt.NDArray):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ConstantEstimator(PDDEstimator):
    def __init__(self, output: npt.NDArray):
        super().__init__(None, None)
        self.output = output

    def fit(self, coords: npt.NDArray, partial_dependence: npt.NDArray):
        raise NotImplementedError

    def __call__(self, data: npt.NDArray):
        return np.tile(self.output, (data.shape[0], 1))


class TreeEstimator(PDDEstimator):
    def __init__(self, categories: Categories, feature_subset: FeatureSubset):
        super().__init__(categories, feature_subset)
        self.tree: Optional[DecisionTreeRegressor] = None

    def fit(self, coords: npt.NDArray, partial_dependence: npt.NDArray):
        self.tree = DecisionTreeRegressor()
        self.tree.fit(coords, partial_dependence)

    def __call__(self, data: npt.NDArray):
        result = self.tree.predict(data)
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result


class ForestEstimator(PDDEstimator):
    def __init__(self, categories: Categories, feature_subset: FeatureSubset):
        super().__init__(categories, feature_subset)
        self.forest: Optional[RandomForestRegressor] = None

    def fit(self, coords: npt.NDArray, partial_dependence: npt.NDArray):
        self.forest = RandomForestRegressor()
        self.forest.fit(coords, partial_dependence)

    def __call__(self, data: npt.NDArray):
        result = self.forest.predict(data)
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)
        return result


class KNNEstimator(PDDEstimator):
    def __init__(self, categories: Categories, feature_subset: FeatureSubset, k=3):
        super().__init__(categories, feature_subset)
        self.k = k
        self.knn: Optional[KNeighborsRegressor] = None

    def _preprocess(self, data: npt.NDArray):
        columns = []
        for i, feat in enumerate(self.feature_subset):
            if feat not in self.categories:
                columns.append(data[:, i].reshape(-1, 1))
            else:
                ohe = OneHotEncoder(categories=[self.categories[feat]], sparse=False)
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
