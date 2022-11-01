import argparse
import time
from experiments.util.datasets import _get_ds_metadata
import json
import os
import pickle
from pddshap import PartialDependenceDecomposition, RandomSubsampleGenerator
import pandas as pd
import numpy as np


def _convert_dtypes(df):
    int_cols = {c: np.int8 for c in df.select_dtypes(include="int64").columns}
    float_cols = {c: np.float32 for c in df.select_dtypes(include="float64").columns}
    return df.astype({**int_cols, **float_cols})


_DESC = """
This script takes an experiment directory (containing output from prerequisites.py), a maximum value for
max_dim, and fixed values for epsilon, the estimator type, and project (boolean).

It will then train a PDDecomposition with max_dim varying between 1 and max_value and generate Shapley values
using the background and test sets in the experiment directory for each decomposition. These Shapley values
are saved to disk in a subdirectory of the experiment directory (exp_name), along with a meta.json file containing
the hyperparameters and runtime information (training + inference).
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESC,
                                     epilog="Example usage:\n\texperiment.py out/abalone varexp_95 --variance-explained 0.95 --estimator knn -k 3")
    parser.add_argument("exp_dir", type=str, help="Directory where the output of prerequisites.py was saved")
    parser.add_argument("exp_name", type=str, help="Name of current experiment. Also name of subdirectory to save the results to.")
    # TODO max-cardinality, variance-explained, project are unused
    parser.add_argument("--max-cardinality", type=int, default=None, help="Maximum cardinality of interactions.")
    parser.add_argument("--coe-threshold", type=float, default=None, help="Cost of Exclusion threshold for deciding if a component should be included.")
    parser.add_argument("--project", action="store_true", help="If set, use orthogonal projection to force the resulting Shapley values to adhere to the completeness axiom.")
    parser.add_argument("--estimator", type=str, choices=["tree", "forest", "knn"], default="tree", help="Model to use to estimate PDD components")
    parser.add_argument("-k", type=int, default=None, help="Parameter k for KNN estimator")
    args = parser.parse_args()

    subdir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
    else:
        raise ValueError(f"Directory {subdir} already exists.")

    with open(os.path.join(args.exp_dir, "meta.json")) as fp:
        exp_meta = json.load(fp)

    print("Loading data and metadata...")
    ds_meta = _get_ds_metadata(exp_meta["dataset"])
    with open(os.path.join(args.exp_dir, "model.pkl"), "rb") as fp:
        model = pickle.load(fp)
    pred_fn = model.predict_proba if ds_meta["pred_type"] == "classification" else model.predict

    X_bg = pd.read_csv(os.path.join(args.exp_dir, "X_bg.csv"))
    X_test = pd.read_csv(os.path.join(args.exp_dir, "X_test.csv"))
    X_bg = _convert_dtypes(X_bg)
    X_test = _convert_dtypes(X_test)
    print("Done.")

    pdd_meta = {
        "coe_threshold": args.coe_threshold,
        "estimator": args.estimator,
        "project": args.project,
        "runtime": {}
    }

    est_kwargs = {}
    if args.k is not None:
        est_kwargs["k"] = args.k

    decomposition = PartialDependenceDecomposition(pred_fn, RandomSubsampleGenerator(), args.estimator, est_kwargs)

    start_t = time.time()
    decomposition.fit(X_bg, args.max_cardinality, args.coe_threshold)
    end_t = time.time()
    fit_time = end_t - start_t

    start_t = time.time()
    values = decomposition.shapley_values(X_test, args.project)
    end_t = time.time()
    infer_time = end_t - start_t
    pdd_meta["runtime"] = {
        "train": fit_time,
        "inference": infer_time
    }
    output = decomposition(X_test)

    np.save(os.path.join(subdir, "values.npy"), values)
    np.save(os.path.join(subdir, "output.npy"), output)

    with open(os.path.join(subdir, "meta.json"), "w") as fp:
        json.dump(pdd_meta, fp)