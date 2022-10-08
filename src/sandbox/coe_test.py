from pddshap.coe import compute_significant_subsets, cost_of_exclusion, cost_of_exclusion_alt
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
    num_features = 6
    num_iterations = 100
    mp = MultilinearPolynomial(num_features, coefficients={
        (): 10,
        (1,2,3): 5.
    })
    data = np.random.normal(size=(1000, num_features))
    print(np.var(mp(data)))

    for key in [(0,), (4,), (0,4), (0,4,5), (1,), (1,2), (1,2,3)]:
    #for key in [(0,4,5)]:
        print(key)
        coes = [cost_of_exclusion_alt(data, mp, key) for _ in range(num_iterations)]
        print(f"\t{np.average(coes):.3f}")
        print()

    """
    significant_subsets = compute_significant_subsets(data, mp, num_iterations=1, threshold=1e-10)

    coefficients = mp.coefficients.keys()
    true_subsets = []
    for coef in coefficients:
        subsets = list(powerset(coef))
        for subset in subsets:
            if tuple(sorted(subset)) not in true_subsets:
                true_subsets.append(tuple(sorted(subset)))

    tp, fp, fn = 0, 0, 0
    significant_subsets_flattened = []
    for key in significant_subsets:
        significant_subsets_flattened += significant_subsets[key]

    for sig_subset in significant_subsets_flattened:
        if sig_subset in true_subsets:
            tp += 1
        else:
            fp += 1
    for subset in true_subsets:
        if subset not in significant_subsets_flattened and len(subset) > 0:
            fn += 1

    print(f"Precision: {tp / (tp + fp):.2f}")
    print(f"Recall: {tp / (tp + fn):.2f}")
    """