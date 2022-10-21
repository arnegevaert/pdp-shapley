from typing import Collection
from numpy import typing as npt
import numpy as np


class FeatureSubset:
    def __init__(self, features: Collection[int] = None):
        self._features = frozenset(features) if features is not None else frozenset()

    def project(self, data: npt.NDArray, values: npt.NDArray) -> npt.NDArray:
        """
        Projects the rows in data to the hyperplane defined by the values.
        In practice, this just overwrites the columns corresponding to this feature subset
        using the values given by values.
        :param data: Data to be projected. Shape: (-1, total_num_features)
        :param values: Values to project the data to. Shape: (total_num_features,)
        :return: Projected data. Shape: (-1, total_num_features)
        """
        # Check shapes
        if len(data.shape) != 2:
            raise ValueError("Invalid shape: data:", data.shape)
        if len(values.shape) != 1:
            raise ValueError("Invalid shape: values:", values.shape)
        data_copy = np.copy(data)
        data_copy[:, sorted(tuple(self._features))] = values[sorted(tuple(self._features))]
        return data_copy

    def get_columns(self, data: npt.NDArray) -> npt.NDArray:
        """
        Extracts the columns in data that correspond to this feature subset.
        :param data: Data from which to extract columns. Shape: (-1, total_num_features)
        :return: Extracted columns. Shape: (-1, len(self.features))
        """
        if len(data.shape) != 2:
            raise ValueError("Invalid shape: data:", data.shape)
        return data[:, sorted(tuple(self._features))]

    def __contains__(self, item):
        return item in self._features

    def __len__(self):
        return len(self._features)

    def __iter__(self):
        return iter(sorted(tuple(self._features)))

    def __repr__(self):
        return "FeatureSubset(" + ", ".join(self.features) + ")"

    def __hash__(self):
        return hash(self._features)

    def __eq__(self, other):
        return type(other) == FeatureSubset and other.features == self.features

    @property
    def features(self):
        return sorted(tuple(self._features))
