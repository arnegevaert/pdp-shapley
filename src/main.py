import shap
import numpy as np
from pddshap import PDDShapleySampler, RandomSubsampleGenerator
from util import datasets, report
import argparse


_DS_DICT = {
    "superconductor": {"name": "superconduct", "type": "regression", "num_outputs": 1},
    "credit": {"name": "credit-g", "type": "classification", "num_outputs": 2},
    "adult": {"type": "classification", "num_outputs": 2}
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, choices=["adult"] + list(_DS_DICT.keys()), default="adult")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=42)

    if args.dataset == "adult":
        X_train, X_test, y_train, y_test, predict_fn = datasets.get_adult()
    else:
        X_train, X_test, y_train, y_test, predict_fn = datasets.get_openml(_DS_DICT[args.dataset])

    
    X_bg = np.copy(X_train)
    rng.shuffle(X_bg)

    with open("../data/adult.npy", "rb") as fp:
        sampling_values = np.load(fp)
    
    #X_test = X_test[:3, :]
    #sampling_values = report.report_time(sampling, "Computing Shapley values via sampling...")

    def sampling():
        explainer = shap.explainers.Sampling(predict_fn, X_bg[:100])
        sampling_values = np.array(explainer.shap_values(X_test)).transpose((1,2,0))
        return sampling_values

    def pdp():
        explainer = PDDShapleySampler(predict_fn, X_bg[:100], num_outputs=_DS_DICT[args.dataset]["num_outputs"], max_dim=4, eps=0.01,
                                      coordinate_generator=RandomSubsampleGenerator(), estimator_type="forest")
        pdp_values = explainer.estimate_shapley_values(X_test)
        return pdp_values

    def permutation():
        #med = np.median(X_train, axis=0).reshape((1,X_train.shape[1]))
        explainer = shap.Explainer(predict_fn, X_bg[:100])
        permutation_values = explainer(X_test)
        return permutation_values

    
    pdp_values = report.report_time(pdp, "Computing Shapley values via PDP...")
    report.report_metrics(pdp_values, sampling_values)
    #report.plot_metrics(pdp_values, sampling_values)

    #permutation_values = report.report_time(permutation, "Computing Shapley values via PermutationExplainer...")
    #report.report_metrics(permutation_values, sampling_values)