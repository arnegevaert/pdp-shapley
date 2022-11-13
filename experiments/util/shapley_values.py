from shap import KernelExplainer, PermutationExplainer, SamplingExplainer
import pandas as pd
import numpy as np


def compute_shapley_values(pred_fn, X_bg: pd.DataFrame, X: pd.DataFrame, explainer: str):
    if explainer == "permutation":
        explainer = PermutationExplainer(pred_fn, X_bg)
        values = explainer(X).values
        return values if len(values.shape) == 3 else np.expand_dims(values, -1)
    elif explainer == "kernel":
        explainer = KernelExplainer(pred_fn, X_bg)
        values = explainer.shap_values(X)
        if type(values) == list:
            return np.stack(values, axis=-1)
        else:
            return np.expand_dims(values, -1)
    elif explainer == "sampling":
        explainer = SamplingExplainer(pred_fn, X_bg)
        values = explainer(X).values
        return values if len(values.shape) == 3 else np.expand_dims(values, -1)
