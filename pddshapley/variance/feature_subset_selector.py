import numpy as np
from typing import Dict, List, Tuple
from pddshapley.util import Model
from numpy import typing as npt
from collections import defaultdict

from pddshapley.signature import FeatureSubset
from pddshapley.variance import VarianceEstimator, COETracker


class FeatureSubsetSelector:
    """
    Based on Liu et al. 2006: Estimating Mean Dimensionality of Analysis of 
    Variance Decompositions

    TODO this class might be unnecessary at this point, it's just a wrapper
    around VarianceEstimator
    """
    def __init__(self, data: npt.NDArray, model: Model):
        self.data = data
        self.model = model
        self.variance_estimator = VarianceEstimator(
                data, model, 
                lower_sobol_strategy="lower_bound",
                lower_sobol_threshold=0.1)

    def get_significant_feature_sets(self, desired_variance_explained,
                                     max_cardinality):
        """
        Computes all subsets (up to a given cardinality) that should be 
        incorporated in an ANOVA decomposition model in order to explain a given
        fraction of the variance.

        :param desired_variance_explained: Desired fraction of variance modeled
            by the components
        :param max_cardinality: Maximal cardinality of subsets.
        :return: Dictionary containing significant subsets for each 
            cardinality: {int: List[Tuple]}
        """

        # Maps cardinality to included feature sets of that cardinality
        # and their estimated component variance for each output
        result: Dict[int,
                     List[
                         Tuple[FeatureSubset,
                               npt.NDArray]]] = defaultdict(list)
        # Empty set should always be included
        result[0].append((
            FeatureSubset(), 
            np.zeros(self.variance_estimator.num_outputs)))
        # Current fraction of variance explained for each output
        variance_explained = np.zeros(self.variance_estimator.num_outputs)
        num_columns = self.data.shape[1]

        # Keep track of CoE values for candidate components, while taking into
        # account possible multiple outputs
        tracker = COETracker(num_columns, self.variance_estimator.num_outputs)

        # We start with all singleton subsets
        for i in range(num_columns):
            # coe contains CoE for each output
            coe = self.variance_estimator.cost_of_exclusion(FeatureSubset(i))
            tracker.push(FeatureSubset(i), coe)

        # subset_counts contains the number of immediate subsets for each 
        # feature set that have been included.
        # If all immediate subsets of a feature set are included, then that 
        # feature set should be added to the queue.
        subset_counts = defaultdict(lambda: 0)

        # Add subsets in order of decreasing CoE until the desired fraction of
        # variance has been included
        while np.any(tracker.active_outputs) and not tracker.empty():
            # Pop the next feature subset to be modeled, which is the one 
            # having the largest CoE over all active outputs
            feature_subset = tracker.pop()
            print(feature_subset)
            if len(feature_subset) <= max_cardinality:
                # Compute component variance and add it to variance explained
                component_variance = self.variance_estimator.component_variance(
                        feature_subset)
                component_variance = np.maximum(component_variance, 0)
                variance_explained += component_variance
                # Add feature subset + its variance to the result
                result[len(feature_subset)].append(
                        (feature_subset, component_variance))
                # Increment subset_counts for each immediate superset of 
                # feature_subset
                for i in range(num_columns):
                    if i not in feature_subset:
                        superset = FeatureSubset(i, *feature_subset)
                        subset_counts[superset] += 1
                        # If all immediate subsets of superset have been 
                        # included, add superset to queue
                        if subset_counts[superset] == len(superset):
                            coe = self.variance_estimator.cost_of_exclusion(
                                    superset)
                            tracker.push(superset, coe)
            # Refresh active outputs
            tracker.active_outputs = variance_explained < \
                desired_variance_explained
        return result
