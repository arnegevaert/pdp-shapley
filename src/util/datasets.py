from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import balanced_accuracy_score
import shap


def get_adult():
    X, y = shap.datasets.adult()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier()
    _train(knn, X_train, y_train)
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, knn.predict(X_test)):.2f}")
    return X_train, X_test, y_train, y_test, knn.predict_proba

def get_openml(ds_dict):
    data = fetch_openml(name=ds_dict["name"])
    X, y = data.data.values, data.target.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if ds_dict["type"] == "classification":
        knn = KNeighborsClassifier()
        _train(knn, X_train, y_train)
        print(f"Balanced accuracy: {balanced_accuracy_score(y_test, knn.predict(X_test)):.2f}")
        pred_fn = knn.predict_proba
    else:
        knn = KNeighborsRegressor()
        _train(knn, X_train, y_train)
        print(f"R2 score: {knn.score(X_test, y_test)}")
        pred_fn = knn.predict
    return X_train, X_test, y_train, y_test, pred_fn


def _train(model, X_train, y_train):
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training done.")
    return model