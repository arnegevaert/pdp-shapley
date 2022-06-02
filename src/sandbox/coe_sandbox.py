import numpy as np
from pddshap.coe import COECalculator
import pandas as pd
from itertools import combinations, chain

"""
def model(X):
    return X[:, 0] + \
           X[:, 1] * X[:, 2] + \
           X[:, 3] + \
           X[:, 0] * X[:, 3] * X[:, 4] + \
           X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]
"""

def _strict_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    # Note: this version only returns strict subsets
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


def model(X: pd.DataFrame):
    return X[0] + X[1] * X[2]


if __name__ == "__main__":
    num_features = 3
    mean = np.random.rand(num_features)
    cov = np.diag(np.ones(num_features))
    X = pd.DataFrame(np.random.multivariate_normal(mean, cov, size=1000))
    y = model(X)

    print(np.var(y))

    calculator = COECalculator(X, model, iterations=10)
    for subset in _strict_powerset(range(num_features)):
        coe = calculator(list(subset))
        print(f"{subset}:\t{coe:.3f}")
