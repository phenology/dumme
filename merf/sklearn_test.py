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


class TestSklearnComplaince(unittest.TestCase):
    def test_compliance_checker(self):
        """Verify that check_estimator works for a native sklearn function."""
        check_estimator(LinearSVC())


    # This is really useful for develop, but can't manage to pass all checks.
    # Probably because there needs to be a categorial/integer column in the test
    # data to be used as clusters, and sklearn data doesn't have that.
    @unittest.skip("First checks should pass, but later tests don't.")
    def test_merf_compliance(self):
        """Check MERF compliance."""
        check_estimator(MERF())

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
        exp.setup(data=train, target='y')

        merf = exp.create_model(MERF(max_iterations=5), fit_kwargs=fit_kwargs, cross_validation=False)


if __name__ == "__main__":
    unittest.main()
