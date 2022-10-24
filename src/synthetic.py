from pddshap import PartialDependenceDecomposition, RandomSubsampleGenerator, KMeansGenerator
from util import report
import numpy as np
import shap
from synth import RandomMultilinearPolynomial
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(0)


def pdd_shap():
    decomposition = PartialDependenceDecomposition(model, coordinate_generator=RandomSubsampleGenerator(),
                                                   estimator_type="knn", est_kwargs={"k": 5})
    decomposition.fit(X_train, coe_threshold=1e-3)
    return decomposition, decomposition.shapley_values(X_test, project=True)


def permutation_shap():
    explainer = shap.explainers.Permutation(model, X_train)
    return np.expand_dims(explainer(X_test).values, -1)


if __name__ == "__main__":
    # Generate input data
    num_features = 10
    mean = np.zeros(num_features)
    cov = np.diag(np.ones(num_features))
    X = np.random.multivariate_normal(mean, cov, size=1000).astype(np.float32)

    model = RandomMultilinearPolynomial(num_features, [-1, -1, 5, 3, 1])
    y = model(X)

    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(num_features)])
    # preproc = Preprocessor(X_df, categorical="ordinal")
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.9)

    pdd, pdd_values = report.report_time(pdd_shap, "Using PDD-SHAP...")
    print("Results:")
    report.report_metrics(pdd_values, np.expand_dims(model.shapley_values(X_test.to_numpy()), -1))

    """
    perm_values = report.report_time(permutation_shap, "Using PermutationSampler...")
    print("Results:")
    report.report_metrics(perm_values, np.expand_dims(model.shapley_values(X_test.to_numpy()), -1))
    """