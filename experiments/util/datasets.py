from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import fetch_openml
import pandas as pd
import os
from typing import Tuple

"""
OpenML Benchmarks:
- https://www.openml.org/search?type=benchmark&sort=tasks_included&study_type=task&id=99
- https://www.openml.org/search?type=benchmark&sort=tasks_included&study_type=task&id=269
- https://www.openml.org/search?type=benchmark&sort=tasks_included&study_type=task&id=297
- https://www.openml.org/search?type=benchmark&sort=tasks_included&study_type=task&id=299

UCI:
- http://archive.ics.uci.edu/ml/datasets/pen-based+recognition+of+handwritten+digits
- https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations
- https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
- https://archive.ics.uci.edu/ml/datasets/BlogFeedback
- https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1998+Data
- https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification

"""

_DS_DICT = {
    "adult": {"data_id": 1590, "pred_type": "classification"},
    "credit": {"data_id": 31, "pred_type": "classification"},
    "abalone": {"data_id": 1557, "pred_type": "classification"},
    "digits": {"data_id": 28, "pred_type": "classification"},
    "iris": {"data_id": 61, "pred_type": "classification"},
    "satimage": {"data_id": 182, "pred_type": "classification"},
    "australian": {"data_id": 40981, "pred_type": "classification"},
    "diabetes": {"data_id": 37, "pred_type": "classification"},
    "phoneme": {"data_id": 1489, "pred_type": "classification"},
    "musk": {"data_id": 40666, "pred_type": "classification"},
    "bodyfat": {"data_id": 560, "pred_type": "regression"},
    "superconduct": {"data_id": 43174, "pred_type": "regression"},
    "housing": {"data_id": 43939, "pred_type": "regression"},
    "kin8nm": {"data_id": 189, "pred_type": "regression"}
}


def get_ds_metadata(ds_name):
    return _DS_DICT[ds_name]


def get_valid_datasets():
    return _DS_DICT.keys()


def get_dataset(ds_name, data_dir, download=True, force_download=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    config = _DS_DICT[ds_name]

    ds_dir = os.path.join(data_dir, ds_name)
    if (not os.path.isdir(ds_dir) and download) or force_download:
        # Get dataset from OpenML
        ds = fetch_openml(data_id=config["data_id"])
        # Drop nan rows
        y = ds.target[ds.data.notnull().all(axis=1)].to_numpy()
        df = ds.data.dropna().reset_index(drop=True)
        # Encode labels
        if config["pred_type"] == "classification":
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
        os.makedirs(ds_dir, exist_ok=True)
        df.to_csv(os.path.join(ds_dir, "data.csv"), index=False)
        np.savetxt(os.path.join(ds_dir, "labels.csv"), y)

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
        X_train.to_csv(os.path.join(ds_dir, "X_train.csv"))
        X_test.to_csv(os.path.join(ds_dir, "X_test.csv"))
        y_train.to_csv(os.path.join(ds_dir, "y_train.csv"))
        y_test.to_csv(os.path.join(ds_dir, "y_test.csv"))
        return X_train, X_test, y_train, y_test, config["pred_type"]
    elif os.path.isdir(ds_dir):
        X_train = pd.read_csv(os.path.join(ds_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(ds_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(ds_dir, "y_train.csv"))
        y_test = pd.read_csv(os.path.join(ds_dir, "y_test.csv"))
        return X_train, X_test, y_train, y_test, config["pred_type"]
    else:
        raise ValueError(f"Dataset {ds_name} not found. Set download=True to allow downloading from OpenML.")
