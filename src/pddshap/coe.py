import numpy as np
from itertools import combinations, product
from typing import Tuple, Callable, NewType, List, Dict
from numpy import typing as npt
from collections import defaultdict
from tqdm import tqdm

Model = NewType("Model", Callable[[npt.NDArray], npt.NDArray])


def cost_of_exclusion_alt(data: npt.NDArray, model: Model, feature_set: Tuple) -> float:
    inner_sum = np.zeros(shape=(data.shape[0]))
    shuffled_data = np.copy(data)
    np.random.shuffle(shuffled_data)

    for i in range(len(feature_set) + 1):
        multiplier = 1 if (len(feature_set) - i) % 2 == 0 else -1
        for subset in combinations(feature_set, i):
            term_data = np.copy(shuffled_data)
            term_data[:, subset] = data[:, subset]
            inner_sum += multiplier * model(term_data)

    return np.sum(np.power(inner_sum, 2)) / (data.shape[0] * 2**len(feature_set))



def cost_of_exclusion(data: npt.NDArray, model: Model, subset: Tuple, num_iterations: int = 1) -> float:
    """
    Compute the cost of exclusion for a given subset of features.
    :param data: background dataset used for the estimation (NDArray[num_rows, num_features])
    :param model: model to be approximated (Callable[[NDArray], NDArray])
    :param subset: subset of features to be excluded
    :param num_iterations: number of iterations
    :return: cost of exclusion for the given subset and model (float)
    """
    columns = list(range(data.shape[1]))
    shuffled_data_arrays = [np.copy(data) for _ in range(num_iterations)]
    for arr in shuffled_data_arrays:
        np.random.shuffle(arr)
    model_output = model(data)
    model_variance = np.var(model_output, axis=0)

    invalid_columns = [col for col in subset if col not in columns]
    if len(invalid_columns) > 0:
        raise ValueError(f"Invalid columns in subset {subset}: {invalid_columns}")

    # Build the upper bound ANOVA structure U
    upper_bound_structure = np.array([np.delete(np.arange(data.shape[1]), i) for i in subset])
    # Use an empirical estimate of the restricted ANOVA model to estimate the model output
    estimate = np.zeros_like(model_output)
    for j in range(upper_bound_structure.shape[0]):
        multiplier = 1 if j % 2 == 0 else -1
        # Get all j-way intersections of rows in upper_bound_structure
        # "j-way intersection" is defined as an intersection between j+1 elements
        # Specifically, "0-way intersections" are just elements
        index_subsets = list(combinations(range(upper_bound_structure.shape[0]), j+1))
        for index_subset in index_subsets:
            # Build the intersection from row indices
            intersection = upper_bound_structure[index_subset[0], :]
            for k in range(1, len(index_subset)):
                intersection = np.intersect1d(intersection, upper_bound_structure[index_subset[k], :], assume_unique=True)
            # Replace the corresponding columns of shuffled data with the original data,
            # get output of the model on this partially shuffled data
            intersection_estimate = np.zeros_like(estimate)
            for shuffled_data in shuffled_data_arrays:
                cur_shuffled_data = np.copy(shuffled_data)
                cur_shuffled_data[:, intersection] = data[:, intersection]
                intersection_estimate += multiplier * model(cur_shuffled_data) / num_iterations
            estimate += intersection_estimate
    coe = np.sum(np.power(model_output - estimate, 2)) / (2**len(subset) * data.shape[0])
    return coe


def compute_significant_subsets(data: npt.NDArray, model: Model, threshold: float = 0.01,
                                num_iterations: int = 1) -> Dict[int, List[Tuple]]:
    """
    Compute all significant subsets of features according to the Cost of Exclusion.
    :param data: Background dataset used for computing CoE (NDArray[num_rows, num_features])
    :param model: Model to be approximated (Callable[[NDArray], NDArray])
    :param threshold: CoE threshold to include a given subset
    :param num_iterations: Number of iterations in computation of CoE
    :return: Dictionary of subsets with CoE>threshold. Each entry of the dictionary contains a list of subsets of
        a given cardinality. Example: result[3] = [significant subsets of length 3]
    """
    result = {}
    cardinality = 0
    num_columns = data.shape[1]
    # Candidates are all subsets that might be significant, i.e. all subsets for which all subsubsets are significant
    candidates = [(i,) for i in range(num_columns)]
    pbar = tqdm()
    while len(candidates) > 0:
        cardinality += 1
        # Compute CoE for each of the subsets, include if significant
        result[cardinality] = [subset for subset in candidates
                               if cost_of_exclusion(data, model, subset, num_iterations) > threshold]
        # For each combination of 2 significant subsets, we test if the cardinality of the union is equal to
        # the current cardinality + 1. At the end, all subsets with count[subset] == cardinality * (cardinality + 1)
        # are those subsets for which all subsubsets are significant. These are the candidates for the next iterations.
        counts = defaultdict(lambda: 0)
        for set1, set2 in tqdm(product(result[cardinality], result[cardinality])):
            union = set(set1).union(set(set2))
            if len(union) == cardinality + 1:
                counts[tuple(union)] += 1
        candidates = [subset for subset in counts.keys() if counts[subset] == cardinality * (cardinality + 1)]
        pbar.update(1)
    return result
