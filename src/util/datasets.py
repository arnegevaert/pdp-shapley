from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import fetch_openml
import pandas as pd


def get_dataset_model(openml_args, num_outputs, pred_type):
    # Get dataset from OpenML
    print("Fetching data...")
    ds = fetch_openml(**openml_args)
    # Drop nan rows
    y = ds.target[ds.data.notnull().all(axis=1)].to_numpy()
    df = ds.data.dropna().reset_index(drop=True)
    # Encode labels
    if pred_type == "classification":
        y = LabelEncoder().fit_transform(y)
    else:
        y = StandardScaler().fit_transform(y.reshape(-1, num_outputs))

    col_dfs = []
    for feat_name in df.columns:
        if df.dtypes[feat_name] in ["object", "bool", "category"]:
            encoder = OrdinalEncoder()
            dtype = "int8"
        elif df.dtypes[feat_name] in ["int64", "float64", "int32", "float32"]:
            encoder = StandardScaler()
            dtype = "float32"
        else:
            raise ValueError("Unrecognized dtype in dataframe")
        col_dfs.append(pd.DataFrame(encoder.fit_transform(df[[feat_name]]), columns=[feat_name], dtype=dtype))
    df = pd.concat(col_dfs, axis=1)
    print("Done.")
    print(f"Data shape: {df.shape}")

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    if pred_type == "classification":
        print("Fitting GBC...")
        model = GradientBoostingClassifier(random_state=42)
    else:
        print("Fitting GBR...")
        model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train.to_numpy(), y_train.flatten())
    print("Done.")
    if pred_type == "classification":
        print(f"Balanced accuracy: {balanced_accuracy_score(y_test, model.predict(X_test)):.5f}")
    else:
        print(f"R2: {r2_score(y_test, model.predict(X_test)):.5f}")
    pred_fn = model.predict_proba if type == "classification" else model.predict
    return X_train, X_test, y_train, y_test, pred_fn
