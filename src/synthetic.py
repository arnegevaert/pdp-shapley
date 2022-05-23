from pddshap import PDDShapleySampler, RandomSubsampleGenerator
from util import report
import numpy as np
import shap
from synth import linear_model

np.random.seed(0)


def pdd_shap():
    explainer = PDDShapleySampler(model, X_train, num_outputs=1, max_dim=2,
                                  coordinate_generator=RandomSubsampleGenerator(),
                                  estimator_type="tree")

    print("Computing Shapley values...")
    return explainer.estimate_shapley_values(X_test, avg_output)


def permutation_shap():
    explainer = shap.explainers.Permutation(model, X_train)
    return np.expand_dims(explainer(X_test).values, -1)


def exact_shap():
    explainer = shap.Explainer(model, X_train)
    return np.expand_dims(explainer(X_test).values, -1)


if __name__ == "__main__":
    # Generate input data
    num_features = 10
    mean = np.zeros(num_features)
    cov = np.diag(np.random.random_sample(num_features))
    X_train = np.random.multivariate_normal(mean, cov, size=1000)
    X_test = np.random.multivariate_normal(mean, cov, size=100)

    model = linear_model.RandomLinearModel(num_features=num_features, order=2)
    y = model(X_test)
    shapley_values = model.shapley_values(X_test)
    avg_output = model.beta[0]

    raw_values, pdd_values = report.report_time(pdd_shap, "Using PDD-SHAP...")
    print("Results:")
    report.report_metrics(pdd_values, np.expand_dims(shapley_values, -1))

    perm_values = report.report_time(permutation_shap, "Using PermutationSampler...")
    print("Results:")
    report.report_metrics(perm_values, np.expand_dims(shapley_values, -1))

    """
    exact_values = report.report_time(exact_shap, "Using ExactExplainer...")
    print("Results:")
    report.report_metrics(exact_values, np.expand_dims(shapley_values, -1))
    """

    """
    shapley_values = shapley_values.reshape(100, 10)
    pdd_values = pdd_values.reshape(100, 10)
    unscaled = unscaled.reshape(100, 10)
    pred_diff = model(X_test) - avg_output
    """