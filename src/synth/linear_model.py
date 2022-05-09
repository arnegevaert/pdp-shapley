import numpy as np
from scipy.special import comb
import itertools
from sklearn.preprocessing import PolynomialFeatures


class RandomLinearModel:
    def __init__(self, num_features, order):
        self.num_features = num_features
        self.order = order
        self.polynomial_features = PolynomialFeatures(order, interaction_only=True)
        self.num_parameters = 1 + np.sum([comb(num_features, i, exact=True) for i in range(1, order + 1)])
        # Parameters: bias, univariate, first order interactions, second order interactions, ...
        self.beta = np.random.uniform(low=-1.0, high=1.0, size=self.num_parameters)
        self.combinations = []
        for i in range(self.order + 1):
            self.combinations += list(itertools.combinations(list(range(self.num_features)), r=i))

    def __call__(self, X):
        assert(X.shape[1] == self.num_features)
        # Add column for each interaction of 2 of more variables
        # Skip first 1 + self.num_features values: 1 bias + all univariate effects
        """
        interaction_columns = []
        for interaction in self.combinations[1 + self.num_features:]:
            column = np.prod(X[:, interaction], axis=1)
            interaction_columns.append(column.reshape(-1, 1))
        interaction_columns = np.hstack(interaction_columns)
        X = np.hstack([X, interaction_columns])
        """
        X = self.polynomial_features.fit_transform(X)
        # Add bias column
        #X = np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X])
        return np.matmul(X, self.beta.transpose())

    # TODO assumes 0 mean and diagonal covariance
    def shapley_values(self, X, mean=None, cov=None):
        result = np.zeros_like(X)
        # Enumerate all combinations, skip the empty subset
        for i, subset in enumerate(self.combinations[1:]):
            column = np.prod(X[:, subset], axis=1)
            result[:, subset] += ((self.beta[i+1] * column) / len(subset)).reshape(-1, 1)
        return result
