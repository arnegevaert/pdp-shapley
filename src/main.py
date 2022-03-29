import shap
from scipy.stats import spearmanr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from time import time
from pdp_decomp import PDPShapleySampler


def shap_sampling(model, to_explain, X_bg):
    print("Shapley sampling using SHAP library...")
    start_t = time()
    explainer = shap.explainers.Sampling(model, X_bg)
    values = explainer.shap_values(to_explain)
    end_t = time()
    print(f"Done in {end_t - start_t:.2f} seconds")
    return values



if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    print("Fitting model...")
    X, y = shap.datasets.adult()
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier()
    #knn = LogisticRegression()
    knn.fit(X_train, y_train)
    print("Fitting done.")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, knn.predict(X_test)):.2f}")
    
    X_bg = np.copy(X_train)
    rng.shuffle(X_bg)
    X_bg = X_bg[:100, :]

    #sampling_values = shap_sampling(knn.predict_proba, X_test[:3, :], X_bg)
    #sampling_values = np.stack(sampling_values, axis=)

    print("Computing PDP decomposition...")
    start_t = time()

    explainer = PDPShapleySampler(lambda x: knn.predict_proba(x), X_bg[:100], num_outputs=2, max_dim=1)
    pdp_values = explainer.estimate_shapley_values(X_test[:3, :].reshape(1, -1))
    print(pdp_values.shape)
    #print(sampling_values)

    end_t = time()
    print(f"Done in {end_t - start_t:.2f} seconds")