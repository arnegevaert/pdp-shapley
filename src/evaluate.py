import argparse
import numpy as np
import os
from util.eval import get_corrs, get_r2
import json


def _compare(values1, values2, name1, name2):
    print(f"Comparing {name1} vs {name2}")
    pearson, spearman = get_corrs(values1, values2)
    r2 = get_r2(values1, values2)
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
        with open(os.path.join(args.dir, f"{explainer}.npy"), "rb") as fp:
            shap_values[explainer] = np.load(fp)

    pdd_shap_values = _get_pdd_output(f"{args.exp_name}/values")
    pdd_output = _get_pdd_output(f"{args.exp_name}/output")

    _compare(shap_values["kernel"], shap_values["permutation"], "KernelExplainer", "PermutationExplainer")
    _compare(shap_values["sampling"], shap_values["permutation"], "SamplingExplainer", "PermutationExplainer")
    _compare(shap_values["kernel"], shap_values["sampling"], "KernelExplainer", "SamplingExplainer")

    print("#" * 80)

    keys = sorted(pdd_shap_values.keys())
    for key in keys:
        _compare(pdd_shap_values[key], shap_values["sampling"], f"PDD-SHAP: {key}", "SamplingExplainer")

    with open(os.path.join(args.dir, "meta.json")) as fp:
        meta = json.load(fp)
    with open(os.path.join(args.dir, "vary_max_dim", "meta.json")) as fp:
        pdd_meta = json.load(fp)

    print("#" * 80)
    for key in meta["runtime"].keys():
        print(f"{key}: {meta['runtime'][key]:.2f}s")
    for key in pdd_meta["runtime"].keys():
        print(f"PDD-SHAP {key}:")
        print(f"\tTraining: {pdd_meta['runtime'][key]['train']:.2f}s")
        print(f"\tInference: {pdd_meta['runtime'][key]['inference']:.2f}s")
