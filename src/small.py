from pdp_decomp import PDPShapleySampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from time import time
from sklearn.inspection import partial_dependence

if __name__ == "__main__":
    num_feat = 5
    random_state = 42
    X, y = make_classification(n_samples=100,
                               n_features=num_feat,
                               n_informative=num_feat,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    svm = SVC(kernel="poly", degree=2, probability=True, random_state=random_state)
    svm.fit(X, y)

    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, svm.predict(X_test)):.2f}")

    print("Computing PDP decomposition...")
    start_t = time()

    explainer = PDPShapleySampler(lambda x: svm.predict_proba(x)[:,0].reshape(-1, 1), X_train, max_dim=4)
    pdp_values = explainer.estimate_shapley_values(X_test[1, :].reshape(1, -1))

    avg_output = np.average(svm.predict_proba(X_train)[:, 0])
    cur_output = svm.predict_proba(X_test[1, :].reshape(1, -1))[:, 0]
    print(f"Output difference: {(cur_output - avg_output)[0]:.2f}")
    print(f"Sum of Shapley values: {np.sum(pdp_values):.2f}")

    end_t = time()
    print(f"Done in {end_t - start_t:.2f} seconds")