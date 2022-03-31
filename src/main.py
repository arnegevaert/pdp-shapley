import shap
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from time import time
from pdp_decomp import PDPShapleySampler


def get_corrs(values1: np.ndarray, values2: np.ndarray):
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
    return np.average(pearsons), np.average(spearmans)



if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    print("Fitting model...")
    X, y = shap.datasets.adult()
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print("Fitting done.")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, knn.predict(X_test)):.2f}")
    
    X_bg = np.copy(X_train)
    rng.shuffle(X_bg)
    X_bg = X_bg[:100, :]

    print("Loading sampling values...")
    with open("data/values.npy", "rb") as fp:
        sampling_values = np.load(fp)
    print("Done.")

    print("Computing PDP decomposition...")
    start_t = time()

    explainer = PDPShapleySampler(knn.predict_proba, X_bg[:100], num_outputs=2, max_dim=2)

    end_t = time()
    print(f"Done in {end_t - start_t:.2f} seconds")


    print("Computing Shapley values via PDP...")
    start_t = time()
    pdp_values = explainer.estimate_shapley_values(X_test)
    end_t = time()
    print(f"Done in {end_t - start_t:.2f} seconds")

    pearson, spearman = get_corrs(pdp_values, sampling_values)
    print("Correlations:")
    print(f"\tPearson: {pearson}")
    print(f"\tSpearman: {spearman}")
    
    print("Computing Shapley values via PermutationExplainer...")
    start_t = time()
    med = np.median(X_train, axis=0).reshape((1,X_train.shape[1]))
    explainer = shap.Explainer(knn.predict_proba, med)
    permutation_values = explainer(X_test).values
    end_t = time()
    print(f"Done in {end_t - start_t:.2f} seconds")
    
    pearson, spearman = get_corrs(permutation_values, sampling_values)
    print("Correlations:")
    print(f"\tPearson: {pearson}")
    print(f"\tSpearman: {spearman}")