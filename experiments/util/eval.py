from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn import metrics


def correlations(values1: np.ndarray, values2: np.ndarray):
    # Shape: [num_samples, num_features, num_outputs]
    if values1.shape != values2.shape:
        raise ValueError(f"Shapes don't match: {values1.shape}, {values2.shape}")
    pearsons = []
    spearmans = []
    for i in range(values1.shape[0]):
        for j in range(values2.shape[2]):
            pearson = pearsonr(values1[i, :, j], values2[i, :, j])[0]
            if not np.isnan(pearson):
                pearsons.append(pearson)
            spearman = spearmanr(values1[i, :, j], values2[i, :, j])[0]
            if not np.isnan(spearman):
                spearmans.append(spearman)
    return np.array(pearsons), np.array(spearmans)


def r2_score(values: np.ndarray, true_values: np.ndarray):
    if values.shape != true_values.shape:
        raise ValueError(f"Shapes don't match: {values.shape}, {true_values.shape}")

    r2_values = []
    for i in range(true_values.shape[2]):
        r2_values.append(metrics.r2_score(true_values[:, :, i].flatten(), values[:, :, i].flatten()))
    return np.array(r2_values)


def print_metrics(values: np.ndarray, true_values: np.ndarray, name1: str = "method 1", name2: str = "method 2"):
    print(f"Comparing {name1} vs {name2}")
    pearson, spearman = correlations(values, true_values)
    r2 = r2_score(values, true_values)
    print(f"\tPearson correlation: {np.average(pearson):.2f}")
    print(f"\tSpearman correlation: {np.average(spearman):.2f}")
    print(f"\tR2: {np.average(r2):.2f}")

