from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import fetch_openml
from pddshap.preprocessor import Preprocessor


def get_dataset_model(openml_args):
    # Get dataset from OpenML
    ds = fetch_openml(**openml_args)
    # Drop nan rows
    y = ds.target[ds.data.notnull().all(axis=1)]
    df = ds.data.dropna().reset_index(drop=True)
    # Encode labels
    y = LabelEncoder().fit_transform(y)

    preproc = Preprocessor(df)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(preproc(X_train), y_train)
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, model.predict(preproc(X_test))):.5f}")
    return X_train, X_test, y_train, y_test, model.predict_proba, preproc
