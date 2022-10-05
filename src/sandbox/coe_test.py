from pddshap.coe import COECalculator
from synth.random_multilinear_polynomial import RandomMultilinearPolynomial
import numpy as np
from itertools import combinations, chain
from tqdm import tqdm


def _strict_powerset(iterable):
    "_strict_powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    # Note: this version only returns strict subsets
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


if __name__ == "__main__":
    num_features = 7
    order_terms = [-1, 0.5, 0.1, 0.01]
    rmp = RandomMultilinearPolynomial(num_features, order_terms)
    data = np.random.normal(size=(1000, num_features))
    coe_calculator = COECalculator(data, rmp, iterations=10)
    subsets = []
    for subset in tqdm(_strict_powerset(range(num_features))):
        coe = coe_calculator(list(subset))
        if coe > 0.01:
            subsets.append(subset)

    tp, fp, fn = 0, 0, 0
    for subset in subsets:
        if subset in rmp.coefficients.keys():
            tp += 1
        else:
            fp += 1
    for subset in rmp.coefficients.keys():
        if subset not in subsets and len(subset) > 0:
            fn += 1

    print(f"Precision: {tp / (tp + fp):.2f}")
    print(f"Recall: {tp / (tp + fn):.2f}")
