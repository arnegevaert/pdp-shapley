import numpy as np
from synth import linear_model
np.random.seed(0)
from pdp_decomp import PDPShapleySampler
from util import report


if __name__ == "__main__":
    # Generate input data
    size = 10000
    mean = [0., 0.]
    cov = np.diag([1., 2.])
    X = np.random.multivariate_normal(mean, cov, size=size)

    model = linear_model.RandomLinearModel(num_features=2, order=2)
    y = model(X)
    shapley_values = model.shapley_values(X)

    explainer = PDPShapleySampler(model, X, num_outputs=1, max_dim=2)
    pdd_values = explainer.estimate_shapley_values(X)

    report.report_metrics(pdd_values, np.expand_dims(shapley_values, -1))