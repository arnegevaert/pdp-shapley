import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Callable


class COECalculator:
    def __init__(self, data: pd.DataFrame, model: Callable[[pd.DataFrame], np.ndarray], preprocessor, iterations=1):
        self.data: pd.DataFrame = data.reset_index(drop=True)
        self.shuffled_dfs = [data.sample(frac=1).reset_index(drop=True) for _ in range(iterations)]
        self.model = model
        self.preprocessor = preprocessor
        self.y = self.model(self.preprocessor(self.data))
        self.var = np.var(self.y, axis=0)
        self.iterations = iterations

    def __call__(self, subset: List[str]):
        # Build the upper bound ANOVA structure U
        columns = list(self.data.columns)
        # TODO does using python lists here introduce an overhead?
        if any([col not in columns for col in subset]):
            raise ValueError(f"Invalid columns in subset: {subset}")
        U = [[col for col in columns if col != subset_col] for subset_col in subset]
        # We use an empirical estimate of the restricted ANOVA model G_U to estimate y
        estimate = np.zeros_like(self.y)
        for j in range(len(U)):
            multiplier = 1 if j % 2 == 0 else -1
            # Get all i-way intersections
            intersect_combs = combinations(range(len(U)), j + 1)
            for intersect_comb in intersect_combs:
                intersection = U[intersect_comb[0]]
                for k in range(1, len(intersect_comb)):
                    intersection = [col for col in intersection if col in U[intersect_comb[k]]]
                # Replace the relevant columns of shuffled data with the original data
                cur_estimate = np.zeros_like(estimate)
                for df in self.shuffled_dfs:
                    shuffled_replaced = df.copy(deep=True)
                    shuffled_replaced[intersection] = self.data[intersection]
                    # Add term to estimate
                    cur_estimate += multiplier * self.model(self.preprocessor(shuffled_replaced))
                estimate += cur_estimate / self.iterations
        return np.sum(np.power((self.y - estimate), 2)) / (self.var * self.data.shape[0])


def cost_of_exclusion_np(data, model, iterations=10):
    num_features = data.shape[1]
    result = {}
    # We shuffle the data to produce the x_{r(i)} from the paper
    # Note that the same shuffle needs to be used for all iterations
    # to ensure monotonicity
    shuffled_arrays = [np.copy(data) for i in range(iterations)]
    for shuffled in shuffled_arrays:
        np.random.shuffle(shuffled)
    for i in range(1, num_features + 1):
        subset_combs = combinations(list(range(num_features)), i)
        for subset in subset_combs:
            subset = list(subset)
            # Build the upper bound ANOVA structure U
            U = np.array([np.delete(np.arange(data.shape[1]), i) for i in subset])
            # We use an empirical estimate of the restricted ANOVA model G_U to estimate y
            y = model(data)
            estimate = np.zeros_like(y)
            for j in range(U.shape[0]):
                multiplier = 1 if j % 2 == 0 else -1
                # Get all i-way intersections

                intersect_combs = combinations(range(U.shape[0]), j+1)
                for intersect_comb in intersect_combs:
                    # Note: an intersection can only be empty if subset is the full feature set.
                    # In that case, we shouldn't be computing the cost of exclusion (and an error will be raised).
                    intersection = U[intersect_comb[0], :]
                    for k in range(1, len(intersect_comb)):
                        intersection = np.intersect1d(intersection, U[intersect_comb[k], :], assume_unique=False)
                    # Replace the relevant columns of shuffled data with the original data
                    cur_estimate = np.zeros_like(estimate)
                    for shuffled in shuffled_arrays:
                        shuffled_replaced = np.copy(shuffled)
                        shuffled_replaced[:, intersection] = data[:, intersection]
                        # Add term to estimate
                        cur_estimate += multiplier * model(shuffled_replaced)
                    estimate += cur_estimate / iterations
            result[tuple(subset)] = np.sum(np.power((y - estimate), 2)) / (data.shape[0])
    return result
