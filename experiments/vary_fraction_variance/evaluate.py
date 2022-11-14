import argparse
import numpy as np
import os
from experiments.util import eval
import json


def _get_pdd_output(subdir):
    result = {}
    for filename in os.listdir(os.path.join(args.dir, subdir)):
        root, ext = os.path.splitext(filename)
        result[root] = np.load(os.path.join(args.dir, subdir, filename))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    parser.add_argument("exp_name")
    args = parser.parse_args()

    shap_values = {}
    for explainer in ["kernel", "permutation", "sampling"]:
        p = os.path.join(args.dir, f"{explainer}.npy")
        if os.path.isfile(p):
            shap_values[explainer] = np.load(p)

    pdd_shap_values = np.load(os.path.join(args.dir, args.exp_name, "values.npy"))
    pdd_output = np.load(os.path.join(args.dir, args.exp_name, "output.npy"))

    print("#" * 80)

    # TODO compare to each of the available explainers
    #   also compare explainers to each other
    eval.print_metrics(pdd_shap_values, shap_values["permutation"], f"PDD-SHAP", "PermutationExplainer")

    with open(os.path.join(args.dir, "meta.json")) as fp:
        meta = json.load(fp)
    with open(os.path.join(args.dir, args.exp_name, "meta.json")) as fp:
        pdd_meta = json.load(fp)

    # Print runtimes
    print("#" * 80)
    for key in meta["runtime"].keys():
        print(f"{key}: {meta['runtime'][key]:.2f}s")
    print(f"PDD-SHAP:")
    print(f"\tTraining: {pdd_meta['runtime']['train']:.2f}s")
    print(f"\tInference: {pdd_meta['runtime']['inference']:.2f}s")
