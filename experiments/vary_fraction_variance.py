import argparse
import joblib
import glob
from tqdm import tqdm
import time
from experiments.util.datasets import get_ds_metadata
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
This script computes Shapley values using PDD-SHAP for varying fractions of variance explained.
It requires a data directory containing the datasets, models and original shapley values.
This corresponds to the OUT_DIR from compute_shap.py.

Results are saved in a separate directory OUT_DIR.
For each fraction of variance x.yyy and model, the results are saved in OUT_DIR/DATASET_NAME/MODEL_NAME/x.yyy.
For example: for dataset adult, fraction of variance 0.95 and model knn, results are saved in OUT_DIR/adult/knn/095.

For each dataset, runtimes are saved in OUT_DIR/DATASET_NAME/runtimes.csv.
This file contains the runtime for training and inference separately, for each fraction of variance.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESC)
    parser.add_argument("-d", "--data-dir", type=str, default="./data",
                        help="Directory where the output of compute_shap.py was saved.")
    parser.add_argument("-o", "--out-dir", type=str, default="./out",
                        help="Output directory where results should be stored.")
    parser.add_argument("-e", "--variance-explained", type=float, nargs="*", default=0.9,
                        help="The different fractions of variance explained. Default: 0.9."
                             "This argument can take multiple values.")
    parser.add_argument("--datasets", type=str, nargs="*",
                        help="Datasets to use. By default, all datasets in DATA_DIR will be used.")
    parser.add_argument("--estimator", type=str, choices=["tree", "forest", "knn"], default="tree",
                        help="Model to use to estimate PDD components")
    parser.add_argument("-k", type=int, default=None,
                        help="Parameter k for KNN estimator. Ignored if estimator is not knn.")
    args = parser.parse_args()

    # Check arguments for validity and default values
    os.makedirs(args.out_dir, exist_ok=True)
    datasets = args.datasets if args.datasets is not None else os.listdir(args.data_dir)
    for ds in datasets:
        if ds not in os.listdir(args.data_dir):
            raise ValueError(f"Dataset {ds} not found in {args.data_dir}.")
    for var in args.variance_explained:
        if not 0 < var <= 1:
            raise ValueError(f"Illegal value for variance-explained: {var}")

    prog = tqdm(datasets)
    for ds_name in prog:
        ds_out_dir = os.path.join(args.out_dir, ds_name)
        ds_data_dir = os.path.join(args.data_dir, ds_name)
        ds_model_dir = os.path.join(ds_data_dir, "models")
        ds_shap_dir = os.path.join(ds_data_dir, "shap")
        for variance_explained in args.variance_explained:
            # Create output subdirectory
            dirname = '1' if variance_explained == 1.0 else str(variance_explained).replace('.', '')
            var_out_dir = os.path.join(ds_out_dir, dirname)
            os.makedirs(var_out_dir, exist_ok=True)

            # Load background and test datasets
            X_bg = _convert_dtypes(pd.read_csv(os.path.join(ds_shap_dir, "X_bg.csv")))
            X_test = _convert_dtypes(pd.read_csv(os.path.join(ds_shap_dir, "X_test.csv")))

            model_names = [os.path.basename(fn)[:-4] for fn in glob.glob(os.path.join(ds_model_dir, "*.pkl"))]
            for model_name in model_names:
                # Load model
                model = joblib.load(os.path.join(ds_model_dir, f"{model_name}.pkl"))
                # Train PDD-SHAP model
                pass

                # Estimate Shapley values on the test set
                pass

                # Save results to disk and write timings to csv
                pass


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
