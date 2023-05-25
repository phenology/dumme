"""Tests that implementation adheres to sklearn api

https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""
import unittest

from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import LinearSVC
from utils import MERFDataGenerator
from pycaret.regression import RegressionExperiment
from merf import MERF
import numpy as np


class TestSklearnCompliance(unittest.TestCase):
    def test_compliance_checker(self):
        """Verify that check_estimator works for a native sklearn function."""
        check_estimator(LinearSVC())

    def test_merf_compliance(self):
        """Check MERF compliance."""
        exclude_checks = [
            # ...,
        ]

        checks = check_estimator(MERF(), generate_only=True)
        for estimator, check in checks:
            name = check.func.__name__
            with self.subTest(check=name):
                if name in exclude_checks:
                    self.skipTest(f"Skipping {name}.")
                else:
                    check(estimator)

    def test_pycaret_compatible(self):
        """Check if can be used with pycaret."""
        np.random.seed(3187)

        dg = MERFDataGenerator(m=0.6, sigma_b=4.5, sigma_e=1)
        train, _, _, _, _, _ = dg.generate_split_samples([1, 3], [3, 2], [1, 1])

        X_train = train.drop("y", axis=1)
        y_train = train["y"]
        fit_kwargs = dict(cluster_column="cluster", fixed_effects=["X_0", "X_1", "X_2"], random_effects=["Z"])

        # Ensure it works as standalone
        m = MERF(max_iterations=5)
        m.fit(X_train, y_train, **fit_kwargs)

        # Now with pycaret
        exp = RegressionExperiment()
        exp.setup(data=train, target="y")

        exp.create_model(MERF(max_iterations=5), fit_kwargs=fit_kwargs, cross_validation=False)


if __name__ == "__main__":
    unittest.main()
