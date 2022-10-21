import numpy as np
from itertools import combinations, product
from typing import Tuple, Callable, NewType
from numpy import typing as npt
from collections import defaultdict

Model = NewType("Model", Callable[[npt.NDArray], npt.NDArray])


class CostOfExclusionEstimator:
    def __init__(self, data: npt.NDArray, model: Model):
        self.data = data
        self.model = model
        self.shuffled_idx = np.arange(data.shape[0])
        np.random.shuffle(self.shuffled_idx)
        self.model_evaluations = {}
        self.model_evaluations[()] = self._get_model_evaluation(())
        random_output = self.model_evaluations[()]
        if len(random_output.shape) == 1:
            self.model_variance = np.var(random_output)
            self.num_outputs = 1
        else:
            self.model_variance = np.var(random_output, axis=0)
            self.num_outputs = random_output.shape[1]

    def get_significant_feature_sets(self, variance_explained=0.9, max_cardinality=None):
        """
        Computes all subsets (up to a given cardinality) that should be incorporated in an ANOVA decomposition
        model in order to explain a given fraction of the variance.
        :param variance_explained: Desired fraction of variance explained by the selected subsets.
        :param max_cardinality: Maximal cardinality of subsets.
        :return: Dictionary containing significant subsets for each cardinality: {int: List[Tuple]}
        """

        result = {}
        cardinality = 0
        num_columns = self.data.shape[1]
        # Candidates are all subsets for which all subsubsets are significant
        candidates = [(i,) for i in range(num_columns)]
        max_cardinality = max_cardinality if max_cardinality is not None else num_columns
        while len(candidates) > 0 and cardinality < max_cardinality:
            cardinality += 1
            # Compute CoE for each of the subsets, include if significant
            result[cardinality] = [subset for subset in candidates
                                   if self.cost_of_exclusion(subset) > 1 - variance_explained]
            # For each combination of 2 significant subsets, we test if the cardinality of the union is equal to
            # the current cardinality + 1. At the end, all subsets with count[subset] == cardinality * (cardinality + 1)
            # are those subsets for which all subsubsets are significant.
            # These are the candidates for the next iterations.
            counts = defaultdict(lambda: 0)
            for set1, set2 in product(result[cardinality], result[cardinality]):
                union = set(set1).union(set(set2))
                if len(union) == cardinality + 1:
                    counts[tuple(union)] += 1
            candidates = [subset for subset in counts.keys() if counts[subset] == cardinality * (cardinality + 1)]
        return result

    def cost_of_exclusion(self, feature_set: Tuple, relative=True):
        inner_sum = np.zeros(shape=(self.data.shape[0], self.num_outputs))
        for i in range(len(feature_set) + 1):
            for subset in combinations(feature_set, i):
                multiplier = 1 if (len(feature_set) - len(subset)) % 2 == 0 else -1
                inner_sum += multiplier * self._get_model_evaluation(subset)
        coe = np.sum(np.power(inner_sum, 2)) / (self.data.shape[0] * 2**len(feature_set))
        if relative:
            return np.max(coe/self.model_variance)
        return np.max(coe)

    def _get_model_evaluation(self, feature_set: Tuple):
        if feature_set in self.model_evaluations.keys():
            return self.model_evaluations[feature_set]
        else:
            data = self.data[self.shuffled_idx, :]
            data[:, feature_set] = self.data[:, feature_set]
            result = self.model(data)
            if len(result.shape) == 1:
                result = np.expand_dims(result, axis=1)
            self.model_evaluations[feature_set] = result
            return result