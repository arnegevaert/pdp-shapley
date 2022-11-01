import argparse
import numpy as np
import os
from experiments.util.eval import correlations, r2_score
import json


def _compare(values1, values2, name1, name2):
    print(f"Comparing {name1} vs {name2}")
    pearson, spearman = correlations(values1, values2)
    r2 = r2_score(values1, values2)
    print(f"\tPearson correlation: {np.average(pearson):.2f}")
    print(f"\tSpearman correlation: {np.average(spearman):.2f}")
    print(f"\tR2: {np.average(r2):.2f}")


def _get_pdd_output(subdir):
    result = {}
    for filename in os.listdir(os.path.join(args.dir, subdir)):
        root, ext = os.path.splitext(filename)
        with open(os.path.join(args.dir, subdir, filename), "rb") as fp:
            result[root] = np.load(fp)
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

    """
    _compare(shap_values["kernel"], shap_values["permutation"], "KernelExplainer", "PermutationExplainer")
    _compare(shap_values["sampling"], shap_values["permutation"], "SamplingExplainer", "PermutationExplainer")
    _compare(shap_values["kernel"], shap_values["sampling"], "KernelExplainer", "SamplingExplainer")
    """

    print("#" * 80)

    _compare(pdd_shap_values, shap_values["permutation"], f"PDD-SHAP", "PermutationExplainer")

    with open(os.path.join(args.dir, "meta.json")) as fp:
        meta = json.load(fp)
    with open(os.path.join(args.dir, args.exp_name, "meta.json")) as fp:
        pdd_meta = json.load(fp)

    print("#" * 80)
    for key in meta["runtime"].keys():
        print(f"{key}: {meta['runtime'][key]:.2f}s")
    print(f"PDD-SHAP:")
    print(f"\tTraining: {pdd_meta['runtime']['train']:.2f}s")
    print(f"\tInference: {pdd_meta['runtime']['inference']:.2f}s")
