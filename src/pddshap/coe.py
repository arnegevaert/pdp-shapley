import numpy as np
from itertools import combinations
from typing import List, Callable


class COECalculator:
    def __init__(self, data: np.ndarray, model: Callable[[np.ndarray], np.ndarray], iterations=1):
        self.data = data
        self.shuffled_arrays = [np.copy(data) for _ in range(iterations)]
        for arr in self.shuffled_arrays:
            np.random.shuffle(arr)
        self.model = model
        self.y = self.model(self.data)
        self.var = np.var(self.y, axis=0)
        self.iterations = iterations

    def __call__(self, subset: List[int]):
        # Build the upper bound ANOVA structure U
        columns = list(range(self.data.shape[1]))
        # TODO does using python lists here introduce an overhead?
        if any([col not in columns for col in subset]):
            raise ValueError(f"Invalid columns in subset: {subset}")
        U = np.array([np.delete(np.arange(self.data.shape[1]), i) for i in subset])
        # We use an empirical estimate of the restricted ANOVA model G_U to estimate y
        estimate = np.zeros_like(self.y)
        for j in range(len(U)):
            multiplier = 1 if j % 2 == 0 else -1
            # Get all i-way intersections
            intersect_combs = combinations(range(U.shape[0]), j + 1)
            for intersect_comb in intersect_combs:
                intersection = U[intersect_comb[0], :]
                for k in range(1, len(intersect_comb)):
                    intersection = np.intersect1d(intersection, U[intersect_comb[k], :], assume_unique=True)
                # Replace the relevant columns of shuffled data with the original data
                cur_estimate = np.zeros_like(estimate)
                for shuffled in self.shuffled_arrays:
                    shuffled_replaced = np.copy(shuffled)
                    shuffled_replaced[:, intersection] = self.data[:, intersection]
                    # Add term to estimate
                    cur_estimate += multiplier * self.model(shuffled_replaced)
                estimate += cur_estimate / self.iterations
        return np.sum(np.power((self.y - estimate), 2)) / (self.var * self.data.shape[0])