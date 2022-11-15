from pddshap import FeatureSubsetSelector, FeatureSubset
from experiments.synthetic.multilinear_polynomial import MultilinearPolynomial
import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    num_features = 2
    model = MultilinearPolynomial(num_features, coefficients={
        FeatureSubset(): 3.,
        FeatureSubset(0): 2.,
        FeatureSubset(1): .5,
        FeatureSubset(0, 1): .1
    })
    mean = np.zeros(num_features)
    cov = np.diag(np.ones(num_features))
    X = np.random.multivariate_normal(mean, cov, size=1000).astype(np.float32)

    coe_estimator = FeatureSubsetSelector(X, model)
    subsets = coe_estimator.get_significant_feature_sets(desired_variance_explained=0.99, max_cardinality=X.shape[1])
    #subsets = {1: [FeatureSubset(0), FeatureSubset(1)], 2: [FeatureSubset(0,1)]}

    print(model)
    print("Subset: CoE Sobol Var RelVar")
    print("=================")
    for card in subsets.keys():
        for subset in subsets[card]:
            print(f"{subset}: "
                  f"{coe_estimator.cost_of_exclusion(subset, relative=False):.3f}"
                  f"\t{coe_estimator.lower_sobol_index(subset, relative=False):.3f}"
                  f"\t{coe_estimator.component_variance(subset, relative=False):.3f}"
                  f"\t{coe_estimator.component_variance(subset, relative=True):.3f}")
