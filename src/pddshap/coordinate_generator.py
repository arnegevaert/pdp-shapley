import numpy as np
import pandas as pd


class CoordinateGenerator:
    def get_coords(self, X):
        raise NotImplementedError


class EquidistantGridGenerator(CoordinateGenerator):
    def __init__(self, grid_res):
        self.grid_res = grid_res

    def get_coords(self, X):
        # Meshgrid creates coordinate matrices for each feature
        mg = np.meshgrid(*[np.linspace(np.min(X[:, i]), np.max(X[:, i]), self.grid_res) for i in range(X.shape[1])])
        # Convert coordinate matrices to a single matrix containing a row for each grid point
        return np.vstack(list(map(np.ravel, mg))).transpose()


class RandomSubsampleGenerator(CoordinateGenerator):
    def __init__(self, frac=1.0):
        self.frac = frac

    def get_coords(self, X: pd.DataFrame):
        if self.frac < 1.0:
            return X.sample(frac=self.frac)
        return X


# TODO
class KMeansGenerator(CoordinateGenerator):
    def __init__(self):
        pass

    def get_coords(self, X):
        pass
