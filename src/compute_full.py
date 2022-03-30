import argparse
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_loc")
    args = parser.parse_args()

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

    explainer = shap.explainers.Sampling(knn.predict_proba, X_bg)

    sampling_values = np.array(explainer.shap_values(X_test)).transpose((1,2,0))
    with open(args.out_loc, "wb") as fp:
        np.save(fp, sampling_values)