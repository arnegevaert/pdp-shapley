import numpy as np
from scipy import special
import itertools
from typing import Dict, Tuple, List, Union
from numpy import typing
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


class RandomMultilinearPolynomial:
    def __init__(self, num_features: int, order_terms: List[Union[int, float]]):
        """
        Arguments:
            num_features: total amount of input features
            order_terms: List array containing the number of terms of each order.
                Example: order_terms[3] = the number of desired third-order terms.
                If order_terms[k] is -1, then all k-order terms are included.
                If len(order_terms) < num_features, then all entries between len(order_terms) and num_features
                are assumed to be 0.
                Example: if len(order_terms) == 3, we will only have a bias, univariate and bivariate terms.
                If order_terms[k] is a float between 0 and 1, it is interpreted as a relative amount:
                the fraction of all possible interactions.
        """
        self.num_features = num_features
        self.coefficients: Dict[Tuple, float] = {}
        for k in range(len(order_terms)):
            total_num_interactions = special.comb(num_features, k)
            if type(order_terms[k]) == int:
                num_interactions = order_terms[k]
            else:
                num_interactions = int(order_terms[k] * total_num_interactions)
            if num_interactions > total_num_interactions:
                raise ValueError(f"Cannot have more than {total_num_interactions} terms of order {k}")
            elif num_interactions == -1:
                # Add all interactions of order k
                for comb in itertools.combinations(list(range(self.num_features)), k):
                    self.coefficients[comb] = np.random.normal()
            else:
                # Add the required amount of interactions of order k
                # If the number of required terms is more than 90% of the total number of terms,
                # we do this by generating all of them and sampling without replacement.
                # Otherwise, we sample randomly, throwing out duplicates
                # (avoids generating ${n \choose order_terms[k]}$ subsets).
                if num_interactions / total_num_interactions > 0.9:
                    all_subsets = list(itertools.combinations(list(range(self.num_features)), k))
                    indices = np.random.choice(len(all_subsets), size=num_interactions, replace=False)
                    for i in indices:
                        subset = all_subsets[i]
                        self.coefficients[subset] = np.random.normal()
                else:
                    count = 0
                    while count < num_interactions:
                        subset = tuple(np.sort(np.random.choice(self.num_features, size=k, replace=False)))
                        if subset not in self.coefficients.keys():
                            self.coefficients[subset] = np.random.normal()
                            count += 1

    def shapley_values(self, data: typing.NDArray):
        """
        Computes closed form Shapley values for inputs given by data.
        This assumes that the input features are centered around 0 and independently distributed.

        :param data: NDArray, shape: (num_rows, self.num_features)
        :return: NDArray, shape: (num_rows, self.num_features)
        """
        result = np.zeros_like(data)
        for term in self.coefficients:
            if len(term) > 0:
                result[:, term] += (self.coefficients[term] * np.prod(data[:, term], axis=1) / len(term)).reshape(-1, 1)
        return result

    def variance(self):
        """
        Computes the theoretical variance of the polynomial,
        under the assumption that all input features are zero-centered with variance 1.
        :return: The theoretical variance (float)
        """
        result = 0.
        for term in self.coefficients:
            if len(term) > 0:
                result += self.coefficients[term]**2
        return result

    def mean(self):
        """
        Returns the theoretical mean of the polynomial,
        under the assumption that all input features are zero-centered.
        This is just the bias.
        :return: The theoretical mean (float)
        """
        if () in self.coefficients:
            return self.coefficients[()]
        return 0.

    def __call__(self, data: typing.NDArray):
        """
        Computes the output of the multilinear polynomial for the given input data
        :param data: NDArray, shape: (num_rows, self.num_features)
        :return: NDArray, shape: (num_rows)
        """
        assert(data.shape[1] == self.num_features)
        output = np.zeros(shape=(data.shape[0]))
        for term in self.coefficients:
            if len(term) == 0:
                output += self.coefficients[term]  # bias term
            else:
                output += self.coefficients[term] * np.prod(data[:, term], axis=1)
        return output

    def __repr__(self):
        s = ""
        for term in self.coefficients:
            coef = self.coefficients[term]
            if len(s) > 0:
                s += "+ " if coef > 0 else "- "
            s += f"{coef if coef > 0 else -coef:.3f}"
            for feat in term:
                s += f" * x{feat}"
            s += "\n"
        return s
