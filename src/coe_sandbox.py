import numpy as np
from pddshap.coe import cost_of_exclusion

"""
def model(X):
    return X[:, 0] + \
           X[:, 1] * X[:, 2] + \
           X[:, 3] + \
           X[:, 0] * X[:, 3] * X[:, 4] + \
           X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]
"""
def model(X):
    return X[:, 0] + X[:, 1]*X[:, 2]

if __name__ == "__main__":
    num_features = 3
    mean = np.random.rand(num_features)
    cov = np.diag(np.ones(num_features))
    X = np.random.multivariate_normal(mean, cov, size=1000)
    y = model(X)

    print(np.var(y))

    coes = cost_of_exclusion(X, model, iterations=10)
    sorted_coes = sorted(coes.items(), key=lambda x: x[1], reverse=True)
    for comb, coe in sorted_coes:
        print(f"{comb}:\t{coe:.3f}")
    print(coes[(0,)] + coes[(1,)] + coes[(2,)] - coes[(1,2)])
