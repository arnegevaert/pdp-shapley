import sklearn
import shap
import numpy as np


if __name__ == "__main__":
    X,y = shap.datasets.adult()
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X.values, y, test_size=0.2, random_state=7)

    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(X_train, y_train)

    f = lambda x: knn.predict_proba(x)
    med = np.median(X_train, axis=0).reshape((1,X_train.shape[1]))

    explainer = shap.Explainer(f, med)
    shap_values = explainer(X_valid[:3, :])
    print(shap_values.shape)