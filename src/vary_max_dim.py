import argparse
from util.datasets import get_ds_metadata
import json
import os
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    parser.add_argument("--max-value", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--estimator", type=str, choices=["tree", "forest", "knn"])
    args = parser.parse_args()

    with open(os.path.join(args.exp_dir, "meta.json")) as fp:
        exp_meta = json.load(fp)

    ds_meta = get_ds_metadata(exp_meta["dataset"])
    with open(os.path.join(args.exp_dir, "model.pkl"), "rb") as fp:
        model = pickle.load(fp)
    pred_fn = model.predict_proba if ds_meta["pred_type"] == "classification" else model.predict

