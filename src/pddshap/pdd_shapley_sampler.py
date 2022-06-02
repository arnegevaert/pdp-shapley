import numpy as np
from pddshap import PDDecomposition
import pandas as pd


class PDDShapleySampler:
    def __init__(self, model, X_background: pd.DataFrame, num_outputs, max_dim=None, eps=None, coordinate_generator=None,
                 estimator_type="lin_interp", num_bg_samples=None) -> None:
        self.model = model
        self.X_background = X_background
        if num_bg_samples is not None:
            self.X_background = X_background.sample(n=num_bg_samples)
        # TODO this is only used for the orthogonal projection
        # TODO if we stop using the projection, remove this
        self.bg_avg = np.average(self.model(self.X_background), axis=0)

        self.pdp_decomp = PDDecomposition(self.model, coordinate_generator, estimator_type)
        self.pdp_decomp.fit(self.X_background, max_dim, eps)
        self.num_outputs = num_outputs

    # TODO this can be optimized, see linear_model.py
    def estimate_shapley_values(self, X):
        result = []
        pdp_values = self.pdp_decomp(X)
        for col in X.columns:
            # [num_samples, num_outputs]
            # TODO am I double counting a bias here (ANOVA component for the empty set)?
            values_i = np.zeros((X.shape[0], self.num_outputs))
            for feature_subset, values in pdp_values.items():
                if col in feature_subset:
                    values_i += values / len(feature_subset)
            result.append(values_i)
        # [num_samples, num_features, num_outputs]
        raw_values = np.stack(result, axis=1)
        # Orthogonal projection of Shapley values onto hyperplane x_1 + ... + x_d = c
        # where c is the prediction difference
        # pred_diff = (self.model(X) - self.bg_avg).reshape(-1, 1, raw_values.shape[-1])
        # return raw_values, raw_values - (np.sum(raw_values, axis=1, keepdims=True) - pred_diff) / X.shape[1]
        return raw_values
