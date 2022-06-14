import argparse
import os
import json
import numpy as np
from util.report import report_metrics
from scipy.stats import spearmanr


def print_result_summary(exp_dir):
    with open(os.path.join(exp_dir, "config.json")) as fp:
        config = json.load(fp)
    print(f"{config['dataset']}: eps={config['eps']}, dim={config['max_dim']}, est={config['estimator_type']}, proj={config['project']}")

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
        if key == "kernel":
            #shap_values = np.transpose(shap_values, (1,2,0))
            pass
        report_metrics(pdp_values, shap_values)

    with open(os.path.join(exp_dir, "pdp_output.npy"), "rb") as fp:
        surrogate_output = np.load(fp)
    with open(os.path.join(exp_dir, "model_output.npy"), "rb") as fp:
        true_output = np.load(fp)
    absolute_output_error = np.abs(surrogate_output - true_output)
    if len(values["perm"].shape) != 3:
        values["perm"] = np.expand_dims(values["perm"], -1)
    absolute_shap_error = np.average(np.abs(pdp_values - values["perm"]), axis=1)
    print(
        f"Correlation between errors: {spearmanr(absolute_output_error[:, 0], absolute_shap_error[:, 0])[0]:.3f}")

    with open(os.path.join(exp_dir, "timing.json")) as fp:
        timing = json.load(fp)
        print("PDP:")
        print(f"\ttrain time: {timing['pdp']['train_time']:.2f}s")
        print(f"\tgen time: {timing['pdp']['gen_time']:.2f}s")
        print(f"\ttotal time: {timing['pdp']['gen_time'] + timing['pdp']['train_time']:.2f}s")
        print("ALTERNATIVES:")
        for key in timing:
            if key != "pdp":
                print(f"\t{key}: {timing[key]['gen_time']:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    args = parser.parse_args()

    for subexperiment in sorted(os.listdir(args.in_dir)):
        exp_dir = os.path.join(args.in_dir, subexperiment)
        print_result_summary(exp_dir)
        print()
        print("#"*80)
        print()
