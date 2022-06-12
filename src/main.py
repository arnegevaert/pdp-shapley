import shap
import argparse
import numpy as np

from pddshap import PDDShapleySampler, RandomSubsampleGenerator
from util import datasets, report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


_DS_DICT = {
    "adult": {"args": {"name": "adult", "version": 2}, "num_outputs": 2},
    "credit": {"args": {"name": "credit-g"}, "num_outputs": 2},
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, choices=_DS_DICT.keys(), default="adult")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, predict_fn, preproc = datasets.get_dataset_model(_DS_DICT[args.dataset]["args"])
    X_bg = X_train.sample(n=100)

    def sampling():
        explainer = shap.explainers.Sampling(predict_fn, X_bg)
        return np.array(explainer.shap_values(X_test)).transpose((1, 2, 0))


    def pdp():
        explainer = PDDShapleySampler(predict_fn, X_bg, preprocessor=preproc, num_outputs=_DS_DICT[args.dataset]["num_outputs"],
                                      eps=0.,
                                      coordinate_generator=RandomSubsampleGenerator(), estimator_type="tree",
                                      max_dim=1)
        return explainer.estimate_shapley_values(X_test[:10])


    def permutation():
        def ord_encode(df):
            result = df.copy()
            for cat_idx in preproc.cat_idx:
                result[cat_idx] = df[cat_idx].cat.codes
            return result
        # med = np.median(X_train, axis=0).reshape((1,X_train.shape[1]))
        explainer = shap.Explainer(lambda df: predict_fn(preproc(df)), ord_encode(X_bg))
        return explainer(ord_encode(X_test[:10]))


    pdp_values = report.report_time(pdp, "Computing Shapley values via PDP...")
    permutation_values = report.report_time(permutation, "Computing Shapley values via permutation...")
    report.report_metrics(pdp_values, permutation_values.values)
    # report.plot_metrics(pdp_values, sampling_values)

    # permutation_values = report.report_time(permutation, "Computing Shapley values via PermutationExplainer...")
    # report.report_metrics(permutation_values, sampling_values[:10, ...])
