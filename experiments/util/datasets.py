from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import fetch_openml
import pandas as pd


_DS_DICT = {
    "adult": {"data_id": 1590, "pred_type": "classification"},
    "credit": {"data_id": 31, "pred_type": "classification"},
    "superconduct": {"data_id": 43174, "pred_type": "regression"},
    "housing": {"data_id": 43939, "pred_type": "regression"},
    "abalone": {"data_id": 1557, "pred_type": "classification"},
    "digits": {},
    "mnist": {}
}


def _get_ds_metadata(ds_name):
    return _DS_DICT[ds_name]


def _get_valid_datasets():
    return _DS_DICT.keys()


def get_dataset_model(ds_name):
    config = _DS_DICT[ds_name]
    data_id = config["data_id"]
    pred_type = config["pred_type"]
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
        model = GradientBoostingClassifier()
    else:
        model = GradientBoostingRegressor()
    model.fit(X_train.to_numpy(), y_train.flatten())
    return X_train, X_test, y_train, y_test, model, pred_type
