"""
This script takes an experiment directory (containing output from prerequisites.py), a maximum value for
max_dim, and fixed values for epsilon, the estimator type, and project (boolean).

It will then train a PDDecomposition with max_dim varying between 1 and max_value and generate Shapley values
using the background and test sets in the experiment directory for each decomposition. These Shapley values
are saved to disk in a subdirectory of the experiment directory (exp_name), along with a meta.json file containing
the hyperparameters and runtime information (training + inference).

TODO this script should be extended to a general-purpose experiment script.
TODO any experiment (i.e. varying epsilon, varying estimator, etc) would then correspond to a subfolder exp_name
"""

import argparse
import time
from util.datasets import get_ds_metadata
import json
import os
import pickle
from pddshap import PDDecomposition, RandomSubsampleGenerator
import pandas as pd
import numpy as np


def _convert_dtypes(df):
    int_cols = {c: np.int8 for c in df.select_dtypes(include="int64").columns}
    float_cols = {c: np.float32 for c in df.select_dtypes(include="float64").columns}
    return df.astype({**int_cols, **float_cols})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--max-value", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--project", type=bool, default=True)
    parser.add_argument("--estimator", type=str, choices=["tree", "forest", "knn"], default="knn")
    parser.add_argument("-k", type=int, default=None)
    args = parser.parse_args()

    subdir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
        os.makedirs(os.path.join(subdir, "output"))
        os.makedirs(os.path.join(subdir, "values"))
    else:
        raise ValueError(f"Directory {subdir} already exists.")

    with open(os.path.join(args.exp_dir, "meta.json")) as fp:
        exp_meta = json.load(fp)

    print("Loading data and metadata...")
    ds_meta = get_ds_metadata(exp_meta["dataset"])
    with open(os.path.join(args.exp_dir, "model.pkl"), "rb") as fp:
        model = pickle.load(fp)
    pred_fn = model.predict_proba if ds_meta["pred_type"] == "classification" else model.predict

    X_bg = pd.read_csv(os.path.join(args.exp_dir, "X_bg.csv"))
    X_test = pd.read_csv(os.path.join(args.exp_dir, "X_test.csv"))
    X_bg = _convert_dtypes(X_bg)
    X_test = _convert_dtypes(X_test)
    print("Done.")

    pdd_meta = {
        "epsilon": args.epsilon,
        "estimator": args.estimator,
        "project": args.project,
        "runtime": {}
    }

    est_kwargs = {}
    if args.k is not None:
        est_kwargs["k"] = args.k

    for max_dim in range(1, args.max_value + 1):
        print(f"max_dim={max_dim}/{args.max_value}")
        decomposition = PDDecomposition(pred_fn, RandomSubsampleGenerator(), args.estimator, est_kwargs)

        start_t = time.time()
        decomposition.fit(X_bg, max_dim, args.epsilon)
        end_t = time.time()
        fit_time = end_t - start_t

        start_t = time.time()
        values = decomposition.shapley_values(X_test, args.project)
        end_t = time.time()
        infer_time = end_t - start_t
        pdd_meta["runtime"][max_dim] = {
            "train": fit_time,
            "inference": infer_time
        }

        output = decomposition(X_test)

        with open(os.path.join(subdir, "values", f"{max_dim}.npy"), "wb") as fp:
            np.save(fp, values)
        with open(os.path.join(subdir, "output", f"{max_dim}.npy"), "wb") as fp:
            np.save(fp, output)
    with open(os.path.join(subdir, "meta.json"), "w") as fp:
        json.dump(pdd_meta, fp)
