import argparse
from experiments.util.datasets import _get_valid_datasets, get_dataset_model
from experiments.util.shapley_values import compute_shapley_values
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score, r2_score
import pickle
import time
import json


_EXPLAINERS = ["kernel", "permutation", "sampling"]
_DESC = """This script trains a model, extracts a background distribution and test set, and computes Shapley values using existing explainers in the shap library. 
It then saves the trained model, background distribution, test set, shapley values, and metadata to disk at a given location.
These files are necessary to reproduce experiments."""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=_DESC,
        epilog="Example usage:\n\tprerequisites.py out/abalone abalone -n 100 -b 100 -e permutation sampling",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("out_dir", type=str, help="Directory where results should be stored")
    parser.add_argument("dataset", type=str, choices=_get_valid_datasets(), help="The dataset to use")
    parser.add_argument("-e", "--explainers", nargs="*", choices=_EXPLAINERS,
                        help="Explainer(s) to use for computing Shapley values")
    parser.add_argument("-n", "--num_test", type=int, default=100, help="Number of test samples")
    parser.add_argument("-b", "--num_bg", type=int, default=100, help="Number of background samples")
    args = parser.parse_args()

    explainers = args.explainers if args.explainers is not None else _EXPLAINERS

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

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
    with open(os.path.join(args.out_dir, "model.pkl"), "wb") as fp:
        pickle.dump(model, fp)
    with open(os.path.join(args.out_dir, "X_bg.csv"), "w") as fp:
        X_bg.to_csv(fp, index=False)
    with open(os.path.join(args.out_dir, "X_test.csv"), "w") as fp:
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
        np.save(os.path.join(args.out_dir, f"{explainer}.npy"), values)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp)
