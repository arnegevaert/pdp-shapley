"""
This script trains a model, extracts a background distribution and test set,
and computes shapley values using the different explainers in the shap package
for a given dataset.

It then saves the trained model, background distribution, test set,
and shapley values to disk.

These files are necessary to start running experiments with PDDecomposition.
"""

import argparse
from util.datasets import get_valid_datasets, get_dataset_model
from util.shapley_values import compute_shapley_values
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score, r2_score
import pickle
import time
import json


_EXPLAINERS = ["kernel", "permutation", "sampling"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    parser.add_argument("dataset", type=str, choices=get_valid_datasets())
    parser.add_argument("-e", "--explainers", nargs="*", choices=_EXPLAINERS)
    parser.add_argument("-n", "--num_test", type=int, default=100)
    parser.add_argument("-b", "--num_bg", type=int, default=100)
    args = parser.parse_args()

    explainers = args.explainers if args.explainers is not None else _EXPLAINERS

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    print("Getting data and training model...")
    X_train, X_test, y_train, y_test, model, pred_type = get_dataset_model(args.dataset)
    pred_fn = model.predict_proba if pred_type == "classification" else model.predict
    if pred_type == "classification":
        y_pred = np.argmax(pred_fn(X_test.to_numpy()), axis=1)
        print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    else:
        print(f"R2: {r2_score(y_test, pred_fn(X_test.to_numpy())):.3f}")
    X_test_sampled = X_test.sample(n=args.num_test)
    X_bg = X_train.sample(n=args.num_bg)

    print("Saving model and data to disk...")
    with open(os.path.join(args.exp_dir, "model.pkl"), "wb") as fp:
        pickle.dump(model, fp)
    with open(os.path.join(args.exp_dir, "X_bg.csv"), "w") as fp:
        X_bg.to_csv(fp, index=False)
    with open(os.path.join(args.exp_dir, "X_test.csv"), "w") as fp:
        X_test_sampled.to_csv(fp, index=False)

    meta = {
        "dataset": args.dataset,
        "num_test": args.num_test,
        "num_bg": args.num_bg,
        "runtime": {}
    }
    for explainer in explainers:
        print(f"Computing Shapley values using {explainer} explainer...")
        start_t = time.time()
        values = compute_shapley_values(pred_fn, X_bg, X_test_sampled, explainer)
        end_t = time.time()
        meta["runtime"][explainer] = end_t - start_t
        with open(os.path.join(args.exp_dir, f"{explainer}.npy"), "wb") as fp:
            np.save(fp, values)
    with open(os.path.join(args.exp_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp)
