from typing import cast
from pddshapley import PartialDependenceDecomposition
from pddshapley.sampling import RandomSubsampleCollocation
from experiments.util import eval
import numpy as np
from experiments.synthetic.multilinear_polynomial import RandomMultilinearPolynomial
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
    print(model)
    y = model(X)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.9)
    X_train = cast(pd.DataFrame, X_train)
    X_test = cast(pd.DataFrame, X_test)

    true_values = np.expand_dims(model.shapley_values(X_test.to_numpy()), -1)

    # Compute Shapley values using PDD-SHAP
    decomposition = PartialDependenceDecomposition(
            model, 
            collocation_method=RandomSubsampleCollocation(),
            estimator_type="knn", est_kwargs={"k": 3})
            #estimator_type="gp")
    decomposition.fit(X_train, variance_explained=0.9)
    pdd_values = decomposition.shapley_values(X_test, project=False)

    eval.print_metrics(pdd_values, true_values, "PDD-SHAP", "Ground truth")

    print()
    for key in decomposition.components.keys():
        print(key)
