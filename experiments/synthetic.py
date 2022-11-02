from pddshap import PartialDependenceDecomposition, RandomSubsampleGenerator
from experiments.util import eval
import numpy as np
import shap
from experiments.util.multilinear_polynomial import RandomMultilinearPolynomial
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Generate input data
    #num_features = 10
    np.random.seed(0)
    num_features = 2
    mean = np.zeros(num_features)
    cov = np.diag(np.ones(num_features))
    X = np.random.multivariate_normal(mean, cov, size=1000).astype(np.float32)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(num_features)])

    # Create model and compute ground truth Shapley values
    #model = RandomMultilinearPolynomial(num_features, [-1, -1, 5, 3, 1])
    model = RandomMultilinearPolynomial(num_features, [-1, -1, -1])
    y = model(X)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.9)
    true_values = np.expand_dims(model.shapley_values(X_test.to_numpy()), -1)

    # Compute Shapley values using PDD-SHAP
    decomposition = PartialDependenceDecomposition(model, coordinate_generator=RandomSubsampleGenerator(),
                                                   estimator_type="knn", est_kwargs={"k": 5})
    decomposition.fit(X_train, coe_threshold=1e-3)
    pdd_values = decomposition.shapley_values(X_test, project=False)

    print("Using PDD-SHAP:")
    eval.print_metrics(pdd_values, true_values, "PDD-SHAP", "Ground truth")

    """
    # Compute Shapley values using PermutationSampling
    perm_explainer = shap.explainers.Permutation(model, X_train)
    perm_values = np.expand_dims(perm_explainer(X_test).values, -1)
    print("Using PermutationExplainer:")
    report.report_metrics(perm_values, np.expand_dims(model.shapley_values(X_test.to_numpy()), -1))
    """