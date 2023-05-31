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
            "check_fit2d_1feature",  # MERF needs at least 2 columns
            "check_parameters_default_constructible",  # fix (None or lambda) makes code less idiomatic
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
    # unittest.main()

    failing_checks = [
        # "check_supervised_y_2d",
        # "check_dtype_object",
        # "check_estimators_overwrite_params",
        # "check_regressor_data_not_an_array",
        # "check_regressors_int",
        # "check_fit_idempotent",
    ]

    checks = check_estimator(MERF(), generate_only=True)
    for estimator, check in checks:
        name = check.func.__name__

        if name in failing_checks:
            check(estimator)


# check_supervised_y_2d    TypeError: NumPy boolean array indexing assignment requires a 0 or 1-dimensional input, input has 2 dimensions
# check_fit2d_predict1d    Should raise
# check_dtype_object    AssertionError: The error message should contain one of the following patterns: Unknown label type Got Cannot cast ufunc 'slogdet' input from dtype('O') to dtype('float64') with casting rule 'same_kind'
# check_estimators_overwrite_params    AssertionError: Estimator MERF should not change or mutate  the parameter fixed_effects_model from RandomForestRegressor(n_estimators=300, n_jobs=-1) to RandomForestRegressor(n_estimators=300, n_jobs=-1) during fit.
# check_regressor_data_not_an_array   Not equal to tolerance rtol=1e-07, atol=0.01
# check_regressors_int   Not equal to tolerance rtol=1e-07, atol=0.01
# check_fit_idempotent   Not equal to tolerance rtol=1e-07, atol=1e-09


#  test_pycaret_compatible
