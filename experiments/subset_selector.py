from pddshap import CostOfExclusionEstimator
from experiments.util.multilinear_polynomial import RandomMultilinearPolynomial
import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    num_features = 2
    model = RandomMultilinearPolynomial(num_features, [-1, -1, -1])
    mean = np.zeros(num_features)
    cov = np.diag(np.ones(num_features))
    X = np.random.multivariate_normal(mean, cov, size=1000000).astype(np.float32)

    coe_estimator = CostOfExclusionEstimator(X, model)
    subsets = coe_estimator.get_significant_feature_sets()

    print(model)
    for card in range(num_features):
        for subset in subsets[card + 1]:
            print(f"{subset}: {coe_estimator.cost_of_exclusion(subset, relative=False):.3f}\t{coe_estimator.lower_sobol_index(subset, relative=False):.3f}")
