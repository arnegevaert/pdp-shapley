import numpy as np
from itertools import combinations, product
from typing import Callable, NewType, List, Dict
from numpy import typing as npt
from collections import defaultdict
import heapq

from pddshap import FeatureSubset

Model = NewType("Model", Callable[[npt.NDArray], npt.NDArray])


class FeatureSubsetSelector:
    """
    Based on Liu et al. 2006: Estimating Mean Dimensionality of Analysis of Variance Decompositions
    """
    def __init__(self, data: npt.NDArray, model: Model):
        self.data = data
        self.model = model
        self.all_features = np.arange(data.shape[1])
        self.shuffled_idx = np.arange(data.shape[0])
        np.random.shuffle(self.shuffled_idx)
        # Contains outputs of the model on the dataset where different subsets of features are kept while the
        # others are randomized. Used by self.cost_of_exclusion to implement a dynamic programming approach.
        self._model_evaluations = {FeatureSubset(): self._compute_model_evaluation(FeatureSubset())}
        random_output = self._get_model_evaluation(FeatureSubset())
        if len(random_output.shape) == 1:
            self.model_variance = np.var(random_output)
            self.num_outputs = 1
        else:
            self.model_variance = np.var(random_output, axis=0)
            self.num_outputs = random_output.shape[1]

    def get_significant_feature_sets(self, desired_variance_explained, max_cardinality):
        """
        Computes all subsets (up to a given cardinality) that should be incorporated in an ANOVA decomposition
        model in order to explain a given fraction of the variance.
        :param desired_variance_explained: Desired fraction of variance modeled by the components
        :param max_cardinality: Maximal cardinality of subsets.
        :return: Dictionary containing significant subsets for each cardinality: {int: List[Tuple]}
        """

        # Maps cardinality to included feature sets of that cardinality
        result = defaultdict(list)
        # Empty set should always be included
        result[0].append(FeatureSubset())
        # Current fraction of variance explained
        variance_explained = 0.
        num_columns = self.data.shape[1]

        # Priority queue keeping track of to be modeled components based on their CoE
        queue = []

        # We start with all singleton subsets
        for i in range(num_columns):
            heapq.heappush(queue, (-self.cost_of_exclusion(FeatureSubset(i)), FeatureSubset(i)))

        # subset_counts contains the number of immediate subsets for each feature set that have been included
        # If all immediate subsets of a feature set are included, then that feature set should be added to the queue
        subset_counts = defaultdict(lambda: 0)

        # Add subsets in order of decreasing CoE until the desired fraction of variance has been included
        while variance_explained < desired_variance_explained and len(queue) > 0:
            # Get the component with largest CoE value, add it to result
            coe, feature_subset = heapq.heappop(queue)
            if len(feature_subset) <= max_cardinality:
                result[len(feature_subset)].append(feature_subset)
            # Increment subset_counts for each immediate superset of feature_subset
            for i in range(num_columns):
                if i not in feature_subset:
                    superset = FeatureSubset(i, *feature_subset)
                    subset_counts[superset] += 1
                    # If all immediate subsets of superset have been included, add superset to queue
                    if subset_counts[superset] == len(superset):
                        heapq.heappush(queue, (-self.cost_of_exclusion(superset), superset))
            # Increase variance explained
            variance_explained += self.component_variance(feature_subset)
        return dict(result)

    def cost_of_exclusion(self, feature_set: FeatureSubset, relative=True) -> float:
        """
        Estimates cost of exclusion for a given feature subset.
        If the model has multiple outputs, returns the maximal CoE value.
        Based on eq. (11) in Liu et al. 2006.
        :param feature_set: The feature subset to compute CoE for.
        :param relative: If true, CoE is returned as a fraction of total variance.
        :return: Estimated CoE.
        """
        inner_sum = np.zeros(shape=(self.data.shape[0], self.num_outputs))
        for i in range(len(feature_set) + 1):
            for subset in combinations(feature_set, i):
                multiplier = 1 if (len(feature_set) - len(subset)) % 2 == 0 else -1
                inner_sum += multiplier * self._get_model_evaluation(FeatureSubset(*subset))
        coe = np.sum(np.power(inner_sum, 2)) / (self.data.shape[0] * 2**len(feature_set))
        if relative:
            return np.max(coe/self.model_variance)
        return np.max(coe)

    def lower_sobol_index(self, feature_set: FeatureSubset, relative=True) -> float:
        r"""
        Estimates the lower Sobol' index :math:`\underline{\tau}` for a given feature subset.
        Based on Sobol', 1993: Sensitivity Estimates for Non-Linear Mathematical Models

        :param feature_set: The feature subset to compute :math:`\underline{\tau}` for.
        :param relative: If true, :math:`\underline{\tau}` is returned as a fraction of total variance.
        :return: Estimated lower Sobol' index.
        """
        orig_output = self._get_model_evaluation(FeatureSubset(*self.all_features))
        shuffled_output = self._get_model_evaluation(feature_set)
        integral = np.average((orig_output * shuffled_output), axis=0)
        sobol = integral - np.average(orig_output)**2
        if relative:
            return np.max(sobol/self.model_variance)
        return np.max(sobol)

    def component_variance(self, feature_set: [FeatureSubset], relative=True) -> float:
        r"""
        Uses the lower Sobol' index and the inclusion-exclusion principle to estimate the variance of a given component
        :param feature_set: The feature set corresponding to the component to be measured
        :param relative: If True, result is expressed as a fraction of total variance
        :return: An estimate of the variance of the component
        """
        result = 0.
        for i in range(1,len(feature_set) + 1):
            for subset in combinations(feature_set, i):
                multiplier = 1 if (len(feature_set) - len(subset)) % 2 == 0 else -1
                result += multiplier * self.lower_sobol_index(FeatureSubset(*subset), relative)
        return result

    def _compute_model_evaluation(self, feature_set: FeatureSubset) -> npt.NDArray:
        # Shuffle the data for all indices not in feature_set
        data = self.data[self.shuffled_idx, :]
        keep_idx = list(feature_set)
        data[:, keep_idx] = self.data[:, keep_idx]
        result = self.model(data)
        if len(result.shape) == 1:
            result = np.expand_dims(result, axis=1)
        return result

    def _get_model_evaluation(self, feature_set: FeatureSubset) -> npt.NDArray:
        """
        Returns the model evaluated on the full dataset, where the features in feature_set are kept and the others
        are randomized. This corresponds to $f(x_i^w, z_i^{-w})$ in eq. (11), where $w$ is represented by feature_set.
        :param feature_set: The features to be kept
        :return: Model output on partially shuffled data. Shape: (self.data.shape[0], self.num_outputs)
        """
        if feature_set not in self._model_evaluations.keys():
            self._model_evaluations[feature_set] = self._compute_model_evaluation(feature_set)
        return self._model_evaluations[feature_set]