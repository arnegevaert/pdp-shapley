import numpy as np
from itertools import combinations


def cost_of_exclusion(data, subset, model):
    if len(subset) == data.shape[1]:
        raise ValueError("Cannot compute cost of exclusion for full feature set")
    if len(subset) == 0:
        raise ValueError("Cannot compute cost of exclusion for empty subset")

    # Build the upper bound ANOVA structure U
    y = model(data)
    # We use an empirical estimate of the restricted ANOVA model G_U to estimate y
    estimate = np.zeros_like(y)
    # We shuffle the data to produce the x_{r(i)} from the paper
    shuffled = np.copy(data)
    np.random.shuffle(shuffled)
    U = np.array([np.delete(np.arange(data.shape[1]), i) for i in subset])
    for i in range(U.shape[0] + 1):
        multiplier = 1 if (U.shape[0] - i) % 2 == 0 else -1
        # Get all i-way intersections
        if i == 0:
            # There is only one 0-way intersection: the complete set
            estimate += multiplier * y
        else:
            combs = combinations(range(U.shape[0]), i)
            for comb in combs:
                # Note: an intersection can only be empty if subset is the full feature set.
                # In that case, we shouldn't be computing the cost of exclusion (and an error will be raised).
                intersection = U[comb[0], :]
                for j in range(1, len(comb)):
                    intersection = np.intersect1d(intersection, U[comb[j], :], assume_unique=True)
                # Replace the relevant columns of shuffled data with the original data
                shuffled_replaced = np.copy(shuffled)
                shuffled_replaced[:, intersection] = data[:, intersection]
                # Add term to estimate
                estimate += multiplier * model(shuffled_replaced)
    return np.sum(np.power((y - estimate), 2)) / (2 * data.shape[0])