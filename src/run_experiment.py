import argparse
import numpy as np
import yaml
import json
import itertools
import os
from util import datasets
from sklearn.metrics import balanced_accuracy_score, r2_score
from pddshap import PDDecomposition, RandomSubsampleGenerator
import time
import shap
import logging
import warnings


_DS_DICT = {
    "adult": {"openml_args": {"name": "adult", "version": 2}, "pred_type": "classification"},
    "credit": {"openml_args": {"name": "credit-g"}, "pred_type": "classification"},
    "superconduct": {"openml_args": {"name": "superconduct"}, "pred_type": "regression"},
    "housing": {"openml_args": {"name": 43939}, "pred_type": "regression"},
    "abalone": {"openml_args": {"data_id": 1557}, "pred_type": "classification"}
}


def measure_runtime(func):
    start_t = time.time()
    return_value = func()
    end_t = time.time()
    return return_value, end_t - start_t


def run_experiment(ds_name, eps, max_dim, estimator_type, num_train, num_test, project):
    # Get data and train model
    print("Getting data...")
    X_train, X_test, y_train, y_test, predict_fn = datasets.get_dataset_model(**_DS_DICT[ds_name])
    X_test_sampled = X_test.sample(num_test)
    model_output = predict_fn(X_test_sampled.to_numpy())
    X_bg = X_train.sample(n=num_train)
    print(f"Number of train samples: {X_train.shape[0]}")
    print(f"Number of test samples: {X_test.shape[0]}")

    if _DS_DICT[ds_name]["pred_type"] == "classification":
        print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predict_fn(X_test.to_numpy())):.5f}")
    else:
        print(f"R2: {r2_score(y_test, predict_fn(X_test.to_numpy())):.5f}")

    values = {}
    timing = {}

    # Train PDP surrogate model
    print("PDPSHAP...")
    decomposition = PDDecomposition(predict_fn, RandomSubsampleGenerator(), estimator_type)
    _, pdp_train_time = measure_runtime(lambda: decomposition.fit(X_bg, max_dim, eps))

    # Generate PDP Shapley values
    pdp_values, pdp_gen_time = measure_runtime(lambda: decomposition.shapley_values(X_test_sampled, project))
    values["pdp"] = pdp_values
    timing["pdp"] = {"gen_time": pdp_gen_time, "train_time": pdp_train_time}
    pdp_output = decomposition(X_test_sampled)

    # Sample Shapley values using PermutationSampler
    print("PermutationExplainer...")
    perm_explainer = shap.PermutationExplainer(predict_fn, X_bg.to_numpy())
    perm_values, perm_gen_time = measure_runtime(lambda: perm_explainer(X_test_sampled.to_numpy()))
    values["perm"] = perm_values
    timing["perm"] = {"gen_time": perm_gen_time, "train_time": 0.}

    # Sample Shapley values using KernelExplainer
    print("KernelExplainer...")
    kernel_explainer = shap.KernelExplainer(predict_fn, X_bg.to_numpy())
    kernel_values, kernel_gen_time = measure_runtime(lambda: kernel_explainer.shap_values(X_test_sampled.to_numpy()))
    values["kernel"] = kernel_values
    timing["kernel"] = {"gen_time": kernel_gen_time, "train_time": 0.}

    return values, timing, model_output, pdp_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_config", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    with open(args.experiment_config, 'r') as stream:
        config = yaml.full_load(stream)

    it = list(itertools.product(*config.values()))
    exp_root_dir = os.path.join(args.out_dir, f"{int(time.time())}")
    print(f"Saving results in {exp_root_dir}")
    os.makedirs(exp_root_dir)
    for i, configuration in enumerate(it):
        print(f"EXPERIMENT {i+1} OF {len(it)}")
        values, timing, model_output, pdp_output = run_experiment(*configuration)
        # Save shapley values
        exp_dir = os.path.join(exp_root_dir, f"{i}")
        os.makedirs(exp_dir)
        for key in values:
            with open(os.path.join(exp_dir, f"{key}.npy"), "wb") as fp:
                np.save(fp, values[key])
        with open(os.path.join(exp_dir, "timing.json"), "w") as fp:
            json.dump(timing, fp)
        with open(os.path.join(exp_dir, "pdp_output.npy"), "wb") as fp:
            np.save(fp, pdp_output)
        with open(os.path.join(exp_dir, "model_output.npy"), "wb") as fp:
            np.save(fp, model_output)
        with open(os.path.join(exp_dir, "config.json"), "w") as fp:
            json.dump(dict(zip(
                ["dataset", "eps", "max_dim", "estimator_type", "num_train", "num_test", "project"],
                configuration
            )), fp)
