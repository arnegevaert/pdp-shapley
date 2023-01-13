from typing import Dict, Union, Optional, List, Tuple
from collections import defaultdict
from pddshapley import ConstantPDDComponent, PDDComponent
from pddshapley.util.model import Model
from pddshapley.sampling import CollocationMethod
from pddshapley.signature import FeatureSubset, DataSignature
import numpy as np
from tqdm import tqdm
import pandas as pd
from numpy import typing as npt
from sklearn import cluster

from pddshapley.variance import VarianceEstimator, COETracker


class PartialDependenceDecomposition:
    def __init__(self, model: Model, collocation_method: CollocationMethod,
                 estimator_type: str, est_kwargs=None) -> None:
        self.model = model
        self.components: Dict[FeatureSubset,
                              Union[ConstantPDDComponent, PDDComponent]] = {}
        self.collocation_method = collocation_method
        self.estimator_type = estimator_type
        self.est_kwargs = est_kwargs if est_kwargs is not None else {}
        self.data_signature: Optional[DataSignature] = None

        self.bg_avg = None
        self.num_outputs: int
    
    def _get_significant_feature_sets(self, data: npt.NDArray, model: Model,
                                      desired_variance_explained: float,
                                      max_cardinality: int):
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
        variance_estimator = VarianceEstimator(
                data, model, lower_sobol_strategy="lower_bound",
                lower_sobol_threshold=0.1)
        # Empty set should always be included
        result[0].append((
            FeatureSubset(), 
            np.zeros(variance_estimator.num_outputs)))
        # Current fraction of variance explained for each output
        variance_explained = np.zeros(variance_estimator.num_outputs)
        num_columns = data.shape[1]

        # Keep track of CoE values for candidate components, while taking into
        # account possible multiple outputs
        tracker = COETracker(num_columns, variance_estimator.num_outputs)

        # We start with all singleton subsets
        for i in range(num_columns):
            # coe contains CoE for each output
            coe = variance_estimator.cost_of_exclusion(FeatureSubset(i))
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
                component_variance = variance_estimator.component_variance(
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
                            coe = variance_estimator.cost_of_exclusion(superset)
                            tracker.push(superset, coe)
            # Refresh active outputs
            tracker.active_outputs = variance_explained < \
                desired_variance_explained
        return result

    def fit(self, background_data: pd.DataFrame, 
            max_cardinality: Optional[int] = None,
            variance_explained: float = 1.,
            kmeans: Optional[int] = None) -> None:
        """
        Fit the partial dependence decomposition using a given
        background dataset.

        TODO better documentation here
        """
        self.data_signature = DataSignature(background_data)
        data_np = background_data.to_numpy()
        # Cluster the background distribution if necessary
        if kmeans is not None:
            data_np = cluster.KMeans(n_clusters=kmeans)\
                        .fit(data_np).cluster_centers_
        self.bg_avg = np.average(self.model(data_np), axis=0)

        # Select subsets to be modeled
        max_cardinality = max_cardinality
        if max_cardinality is None:
            max_cardinality = data_np.shape[1]
        
        significant_feature_sets = self._get_significant_feature_sets(
                data_np, self.model, variance_explained, max_cardinality)

        total_sets = 0
        for card in significant_feature_sets.keys():
            total_sets += len(significant_feature_sets[card])

        prog = tqdm(total=total_sets)
        # First, model the empty component
        # TODO why do we include the empty component in significant_feature_sets?
        empty_component = ConstantPDDComponent()
        empty_component.fit(data_np, self.model)
        self.components[FeatureSubset()] = empty_component
        if empty_component.num_outputs is not None:
            self.num_outputs: int = empty_component.num_outputs
        else:
            # This should never happen, but is included for the type system
            raise ValueError("Empty component has no num_outputs")
        prog.update()

        # Model the subsets in order of increasing cardinality
        for card in significant_feature_sets.keys():
            if card != 0:
                card_sets = significant_feature_sets[card]
                for feature_set, component_variance in card_sets:
                    # All subcomponents are necessary to compute the values
                    # for this component
                    subcomponents = {k: v for k, v in self.components.items()
                                     if all(
                                         [feat in feature_set for feat in k])}
                    # TODO use component_variance to see if we need to model
                    # this component
                    component = PDDComponent(feature_set, self.data_signature,
                            self.collocation_method, self.estimator_type,
                            self.est_kwargs)
                    component.fit(data_np, self.model, subcomponents)
                    self.components[feature_set] = component
                    prog.update()

    def __call__(self, data: pd.DataFrame | npt.NDArray):
        """
        Compute the output of the decomposition at the given coordinates
        :param data: The coordinates where we evaluate the model
        :return: The output of the decomposition at each coordinate.
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        pdp_values = self.evaluate(data)
        result = np.zeros(shape=(data.shape[0], self.num_outputs))
        for _, values in pdp_values.items():
            result += values
        return result

    def evaluate(self, data: npt.NDArray) -> Dict[FeatureSubset, npt.NDArray]:
        """
        Evaluate PDP decomposition at all rows in data
        :param data: [num_rows, num_features]
        :return: Dictionary containing each component function value:
            {FeatureSubset, npt.NDArray[num_rows, num_outputs]}
        """
        return {subset: component(data) 
                for subset, component in self.components.items()}

    def shapley_values(self, data: pd.DataFrame | npt.NDArray,
                       project=False) -> npt.NDArray:
        """
        Compute Shapley values for each row in data.
        :param data: DataFrame or NDArray, shape: (num_rows, self.num_features)
        :param project: Boolean value indicating if the results should be 
            orthogonally projected to the hyperplane satisfying the 
            Efficiency axiom.
        :return: NDArray containing Shapley values for each row and each output.
            Shape: (num_rows, self.num_features, num_outputs)
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        pdp_values = self.evaluate(data)

        result = np.zeros(shape=(data.shape[0],
                                 data.shape[1],
                                 self.num_outputs))
        for feature_subset, output_vector in pdp_values.items():
            if len(feature_subset) > 0:
                component_effect = np.expand_dims(output_vector, axis=1)
                features = feature_subset.features
                result[:, features, :] += component_effect / len(feature_subset)
        if project:
            # Orthogonal projection of Shapley values onto hyperplane
            # x_1 + ... + x_d = c where c is the prediction difference
            pred_diff = (self.model(data) - self.bg_avg)
            pred_diff = pred_diff.reshape(-1, 1, result.shape[-1])
            adjustment = (np.sum(result, axis=1, keepdims=True) - pred_diff)
            adjustment /= data.shape[1]
            return result - adjustment
        return result
