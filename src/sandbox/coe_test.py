from pddshap.coe import CostOfExclusionEstimator
from synth.random_multilinear_polynomial import RandomMultilinearPolynomial, MultilinearPolynomial
import numpy as np
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


if __name__ == "__main__":
    """
    num_features = 10
    mp = RandomMultilinearPolynomial(num_features, [-1, 10, 5, 1], coefficient_generator=lambda: 10 + np.random.random())
    """
    num_features = 5
    num_iterations = 1
    mp = MultilinearPolynomial(num_features, coefficients={
        (1,2,3): 5.,
        (0,1,2,3,4): 1.
    })
    data = np.random.normal(size=(100000, num_features))
    coe_est = CostOfExclusionEstimator(data, mp)

    significant_subsets = coe_est.get_significant_subsets(variance_explained=0.9)
    print(significant_subsets)
