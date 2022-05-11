import numpy as np
from pddshap import PDDecomposition


class PDDShapleySampler:
    def __init__(self, model, X_background, num_outputs, max_dim=1, eps=0.05, coordinate_generator=None,
                 estimator_type="lin_interp") -> None:
        self.model = model
        self.X_background = X_background

        self.pdp_decomp = PDDecomposition(self.model, coordinate_generator, estimator_type)
        self.pdp_decomp.fit(self.X_background, max_dim, eps)
        self.num_outputs = num_outputs

    # TODO this can be optimized, see linear_model.py
    def estimate_shapley_values(self, X, avg_output):
        result = []
        pdp_values = self.pdp_decomp(X)
        for i in range(X.shape[1]):
            # [num_samples, num_outputs]
            values_i = np.zeros((X.shape[0], self.num_outputs))
            for feature_subset, values in pdp_values.items():
                if i in feature_subset:
                    values_i += values / len(feature_subset)
            result.append(values_i)
        # [num_samples, num_features, num_outputs]
        raw_values = np.stack(result, axis=1)
        # Orthogonal projection of Shapley values onto hyperplane x_1 + ... + x_d = c
        # where c is the prediction difference
        pred_diff = (self.model(X) - avg_output).reshape(-1, 1, 1)
        return raw_values, raw_values - (np.sum(raw_values, axis=1, keepdims=True) - pred_diff) / X.shape[1]
