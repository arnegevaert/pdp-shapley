import shap
import argparse
import numpy as np

from pddshap import PDDecomposition, RandomSubsampleGenerator
from util import datasets, report


_DS_DICT = {
    "adult": {"openml_args": {"name": "adult", "version": 2}, "num_outputs": 2, "pred_type": "classification"},
    "credit": {"openml_args": {"name": "credit-g"}, "num_outputs": 2, "pred_type": "classification"},
    "superconduct": {"openml_args": {"name": "superconduct"}, "num_outputs": 1, "pred_type": "regression"},
    "housing": {"openml_args": {"name": 43939}, "num_outputs": 1, "pred_type": "regression"},
    "abalone": {"openml_args": {"data_id": 1557}, "num_outputs": 3, "pred_type": "classification"}
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, choices=_DS_DICT.keys(), default="abalone")
    parser.add_argument("-e", "--eps", type=float, default=None)
    parser.add_argument("-m", "--max-dim", type=int, default=3)
    parser.add_argument("--estimator-type", type=str, default="tree")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=100)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, predict_fn = datasets.get_dataset_model(**_DS_DICT[args.dataset])
    X_test_sampled = X_test.sample(args.num_test)
    X_bg = X_train.sample(n=args.num_train)
    print(f"Number of train samples: {X_train.shape[0]}")
    print(f"Number of test samples: {X_test.shape[0]}")

    def sampling():
        explainer = shap.explainers.Sampling(predict_fn, X_bg)
        return np.array(explainer.shap_values(X_test)).transpose((1, 2, 0))


    def pdp():
        decomposition = PDDecomposition(predict_fn, RandomSubsampleGenerator(), args.estimator_type)
        decomposition.fit(X_bg, max_dim=args.max_dim, eps=args.eps)
        return decomposition.shapley_values(X_test_sampled, project=True)


    def permutation():
        # med = np.median(X_train, axis=0).reshape((1,X_train.shape[1]))
        explainer = shap.PermutationExplainer(predict_fn, X_bg.to_numpy())
        return explainer(X_test_sampled.to_numpy())

    pdp_values = report.report_time(pdp, "Computing Shapley values via PDP...")
    permutation_values = report.report_time(permutation, "Computing Shapley values via permutation...")
    permutation_values = permutation_values.values
    if len(permutation_values.shape) < 3:
        permutation_values = np.expand_dims(permutation_values, -1)
    report.report_metrics(pdp_values, permutation_values)
    # report.plot_metrics(pdp_values, sampling_values)

    # permutation_values = report.report_time(permutation, "Computing Shapley values via PermutationExplainer...")
    # report.report_metrics(permutation_values, sampling_values[:10, ...])
