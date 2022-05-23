import numpy as np
from itertools import combinations
from pddshap.coe import cost_of_exclusion


def model(X):
    return X[:, 0] + X[:, 1] * X[:, 2] + X[:, 3] + X[:, 0] * X[:, 3] * X[:, 4]


if __name__ == "__main__":
    num_features = 5
    mean = np.random.rand(num_features)
    cov = np.diag(np.ones(num_features)*5)
    X = np.random.multivariate_normal(mean, cov, size=100)
    y = model(X)

    #X = np.arange(15).reshape(5,3)

    for i in range(1, num_features):
        combs = combinations(list(range(num_features)), i)
        coes = {comb: cost_of_exclusion(X, list(comb), model) / np.var(y) for comb in combs}
        sorted_coes = sorted(coes.items(), key=lambda x: x[1], reverse=True)
        print(f"{i}-WAY INTERACTIONS")
        for comb, coe in sorted_coes:
            print(f"{comb}:\t{coe:.3f}")
        print()