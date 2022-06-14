from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import fetch_openml
import pandas as pd


def get_dataset_model(data_id, pred_type):
    # Get dataset from OpenML
    ds = fetch_openml(data_id=data_id)
    # Drop nan rows
    y = ds.target[ds.data.notnull().all(axis=1)].to_numpy()
    df = ds.data.dropna().reset_index(drop=True)
    # Encode labels
    if pred_type == "classification":
        y = LabelEncoder().fit_transform(y)
    else:
        y = StandardScaler().fit_transform(y.reshape(-1, 1))

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

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    if pred_type == "classification":
        print("Fitting classifier...")
        model = GradientBoostingClassifier()
    else:
        print("Fitting regressor...")
        model = GradientBoostingRegressor()
    model.fit(X_train.to_numpy(), y_train.flatten())
    pred_fn = model.predict_proba if pred_type == "classification" else model.predict
    return X_train, X_test, y_train, y_test, pred_fn
