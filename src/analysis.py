import argparse
import os
import json
import numpy as np
from util.report import report_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    args = parser.parse_args()

    for subexperiment in sorted(os.listdir(args.in_dir)):
        exp_dir = os.path.join(args.in_dir, subexperiment)
        with open(os.path.join(exp_dir, "config.json")) as fp:
            config = json.load(fp)
        for key, value in config.items():
            print(f"{key}: {value}")

        values = {}
        with open(os.path.join(exp_dir, f"pdp.npy"), "rb") as fp:
            pdp_values = np.load(fp)

        for key in ["perm", "kernel"]:
            if f"{key}.npy" in os.listdir(exp_dir):
                with open(os.path.join(exp_dir, f"{key}.npy"), "rb") as fp:
                    values[key] = np.load(fp, allow_pickle=True)

        for key in values.keys():
            print(f"Comparison to {key}:")
            shap_values = np.expand_dims(values[key], -1) if len(values[key].shape) != 3 else values[key]
            report_metrics(pdp_values, shap_values)

        with open(os.path.join(exp_dir, "pdp_output.npy"), "rb") as fp:
            surrogate_output = np.load(fp)
        with open(os.path.join(exp_dir, "model_output.npy"), "rb") as fp:
            true_output = np.load(fp)
        absolute_output_error = np.abs(surrogate_output - true_output)
        absolute_shap_error = np.average(np.abs(pdp_values - values["perm"]), axis=1)
        print(f"Correlation between errors: {np.corrcoef(absolute_output_error[:, 0], absolute_shap_error[:, 0])[0, 1]:.3f}")
