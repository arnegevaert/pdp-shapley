import numpy as np
from synth import linear_model
np.random.seed(0)
from pddshap import PDDShapleySampler, RandomSubsampleGenerator
from util import report


if __name__ == "__main__":
    # Generate input data
    num_features = 10
    mean = np.zeros(num_features)
    cov = np.diag(np.random.random_sample(num_features))
    X_train = np.random.multivariate_normal(mean, cov, size=1000)
    X_test = np.random.multivariate_normal(mean, cov, size=100)

    model = linear_model.RandomLinearModel(num_features=num_features, order=3)
    y = model(X_train)
    shapley_values = model.shapley_values(X_test)

    explainer = PDDShapleySampler(model, X_train, num_outputs=1, max_dim=3, coordinate_generator=RandomSubsampleGenerator(),
                                  estimator_type="tree")

    print("Computing Shapley values...")
    pdd_values = explainer.estimate_shapley_values(X_test)

    print("Computing metrics...")
    report.report_metrics(pdd_values, np.expand_dims(shapley_values, -1))
    diff = (pdd_values[:, :, 0] - shapley_values) / shapley_values