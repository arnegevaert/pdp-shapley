import numpy as np
from itertools import combinations
from typing import Dict
from pddshapley.util import Model
from numpy import typing as npt
from collections import defaultdict

from pddshapley.signature import FeatureSubset


class COETracker:
    """
    Keeps track of CoE values for multiple outputs and which outputs are active
    TODO Might still be able to use heaps here: re-heapify after an output has been set to inactive.
        Only useful if linear search operations in this class turn out to be a bottleneck, which is very unlikely
    """
    def __init__(self, num_columns, num_outputs):
        self.num_columns = num_columns
        self.num_outputs = num_outputs
        # Maps feature subsets to CoE values for each output
        self.subset_coe: Dict[FeatureSubset, npt.NDArray] = {}
        # True if corresponding output is active, False otherwise
        self.active_outputs = np.array([True for _ in range(num_outputs)])

    def push(self, subset: FeatureSubset, coe_values: npt.NDArray):
        """
        Add a feature subset to the data structure
        """
        self.subset_coe[subset] = coe_values

    def pop(self) -> FeatureSubset:
        """
        Retrieve and remove the feature subset with the largest CoE value among the active outputs
        """
        max_subset = None
        max_coe = 0.
        for subset in self.subset_coe.keys():
            coe = np.max(self.subset_coe[subset][self.active_outputs])
            if coe > max_coe:
                max_subset = subset
                max_coe = coe
        del self.subset_coe[max_subset]
        return max_subset

    def empty(self) -> bool:
        return len(self.subset_coe.keys()) == 0


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
        # Current fraction of variance explained for each output
        variance_explained = np.zeros(self.num_outputs)
        num_columns = self.data.shape[1]

        # Keep track of CoE values for candidate components, while taking into account possible multiple outputs
        tracker = COETracker(num_columns, self.num_outputs)

        # We start with all singleton subsets
        for i in range(num_columns):
            # coe contains CoE for each output
            coe = self.cost_of_exclusion(FeatureSubset(i))
            tracker.push(FeatureSubset(i), coe)

        # subset_counts contains the number of immediate subsets for each feature set that have been included
        # If all immediate subsets of a feature set are included, then that feature set should be added to the queue
        subset_counts = defaultdict(lambda: 0)

        # Add subsets in order of decreasing CoE until the desired fraction of variance has been included
        while np.any(tracker.active_outputs) and not tracker.empty():
            # Pop the next feature subset to be modeled, which is the one having the largest CoE over all active outputs
            feature_subset = tracker.pop()
            print(feature_subset)
            if len(feature_subset) <= max_cardinality:
                result[len(feature_subset)].append(feature_subset)
            # Increment subset_counts for each immediate superset of feature_subset
            for i in range(num_columns):
                if i not in feature_subset:
                    superset = FeatureSubset(i, *feature_subset)
                    subset_counts[superset] += 1
                    # If all immediate subsets of superset have been included, add superset to queue
                    if subset_counts[superset] == len(superset):
                        coe = self.cost_of_exclusion(superset)
                        tracker.push(superset, coe)
            # Increase variance explained
            variance_explained += np.maximum(self.component_variance(feature_subset, relative=True), 0)
            print("\t".join([f"{v:.3}" for v in variance_explained]))
            # Refresh active outputs
            tracker.active_outputs = variance_explained < desired_variance_explained
        return dict(result)

    def component_variance(self, feature_set: FeatureSubset, relative=True) -> npt.NDArray:
        r"""
        Uses the lower Sobol' index and the inclusion-exclusion principle to estimate the variance of a given component
        :param feature_set: The feature set corresponding to the component to be measured
        :param relative: If True, result is expressed as a fraction of total variance
        :return: An estimate of the variance of the component
        """
        result = np.zeros(self.num_outputs)
        for i in range(1, len(feature_set) + 1):
            for subset in combinations(feature_set, i):
                multiplier = 1 if (len(feature_set) - len(subset)) % 2 == 0 else -1
                result += multiplier * self.lower_sobol_index(FeatureSubset(*subset), relative)
        return result
