import numpy as np
from pddshap.coe import COECalculator, cost_of_exclusion_np
from itertools import combinations, chain
import time

def model(X):
    return X[:, 0] + \
           X[:, 1] * X[:, 2] + \
           X[:, 3] + \
           X[:, 0] * X[:, 3] * X[:, 4] + \
           X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]

def _strict_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    # Note: this version only returns strict subsets
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

"""
def model(X: np.ndarray):
    return X[:, 0] + X[:, 1] * X[:, 2]
"""

if __name__ == "__main__":
    num_features = 5
    mean = np.random.rand(num_features)
    cov = np.diag(np.ones(num_features))
    X = np.random.multivariate_normal(mean, cov, size=1000)
    y = model(X)

    print(np.var(y))

    start_t = time.time()
    coe_dict = cost_of_exclusion_np(X, model, iterations=1000)
    end_t = time.time()
    print(f"Numpy implementation took {end_t - start_t:.3f}s")

    start_t = time.time()
    calculator = COECalculator(X, model, iterations=1000)
    for subset in _strict_powerset(range(num_features)):
        coe = calculator(list(subset))
        #coe_np = coe_dict[subset]
        #print(f"{subset}:\t{coe:.3f}\t{coe_np:.3f}\t{coe_np/coe:.3f}")
    end_t = time.time()
    print(f"Pure Python implementation took {end_t - start_t:.3f}s")