from shap import KernelExplainer, PermutationExplainer, SamplingExplainer
import pandas as pd
import numpy as np


def compute_shapley_values(pred_fn, X_bg, X, explainer: str):
    if type(X_bg) == pd.DataFrame:
        X_bg = X_bg.to_numpy()
    if type(X) == pd.DataFrame:
        X = X.to_numpy()
    if explainer == "permutation":
        explainer = PermutationExplainer(pred_fn, X_bg)
        return explainer(X).values
    elif explainer == "kernel":
        explainer = KernelExplainer(pred_fn, X_bg)
        return np.transpose(explainer.shap_values(X), (1, 2, 0))
    elif explainer == "sampling":
        explainer = SamplingExplainer(pred_fn, X_bg)
        return explainer(X).values
