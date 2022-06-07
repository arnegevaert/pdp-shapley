import shap
import argparse
import numpy as np

from pddshap import PDDShapleySampler, RandomSubsampleGenerator
from util import datasets, report


_DS_DICT = {
    "adult": {"args": {"name": "adult", "version": 2}, "num_outputs": 2},
    "credit": {"args": {"name": "credit-g"}, "num_outputs": 2},
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, choices=_DS_DICT.keys(), default="adult")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, predict_fn = datasets.get_dataset_model(_DS_DICT[args.dataset]["args"])
    X_bg = X_train.sample(n=100)

    # TODO generalize this somehow
    with open("../data/adult.npy", "rb") as fp:
        sampling_values = np.load(fp)


    def sampling():
        explainer = shap.explainers.Sampling(predict_fn, X_bg)
        return np.array(explainer.shap_values(X_test)).transpose((1, 2, 0))


    def pdp():
        explainer = PDDShapleySampler(predict_fn, X_bg, num_outputs=_DS_DICT[args.dataset]["num_outputs"],
                                      eps=None,
                                      coordinate_generator=RandomSubsampleGenerator(), estimator_type="forest",
                                      max_dim=1)
        return explainer.estimate_shapley_values(X_test)


    def permutation():
        # med = np.median(X_train, axis=0).reshape((1,X_train.shape[1]))
        explainer = shap.Explainer(predict_fn, X_bg)
        permutation_values = explainer(X_test[:10])
        return permutation_values


    pdp_values = report.report_time(pdp, "Computing Shapley values via PDP...")
    report.report_metrics(pdp_values, sampling_values)
    # report.plot_metrics(pdp_values, sampling_values)

    # permutation_values = report.report_time(permutation, "Computing Shapley values via PermutationExplainer...")
    # report.report_metrics(permutation_values, sampling_values[:10, ...])
