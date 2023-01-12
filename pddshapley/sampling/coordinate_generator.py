from abc import abstractmethod
import numpy as np
from numpy import typing as npt
from sklearn import cluster


class CoordinateGenerator:
    @abstractmethod
    def get_coords(self, data: npt.NDArray):
        raise NotImplementedError


class RandomSubsampleGenerator(CoordinateGenerator):
    def __init__(self, frac=1.0):
        self.frac = frac

    def get_coords(self, data: npt.NDArray):
        if self.frac < 1.0:
            indices = np.random.choice(np.arange(data.shape[0]), replace=False,
                                       size=int(data.shape[0] * self.frac))
            result = data[indices, ...]
            if len(result.shape) == 1:
                result = np.expand_dims(result, axis=-1)
            return result
        return data


class KMeansGenerator(CoordinateGenerator):
    def __init__(self, k: int):
        self.k = k

    def get_coords(self, data: npt.NDArray):
        return cluster.KMeans(n_clusters=self.k).fit(data).cluster_centers_
