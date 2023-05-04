"""Tests that implementation adheres to sklearn api

https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""
import unittest

from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import LinearSVC

from merf import MERF


class TestSklearnComplaince(unittest.TestCase):
    def test_compliance_checker(self):
        """Verify that check_estimator works for a native sklearn function."""
        check_estimator(LinearSVC())

    def test_merf_compliance(self):
        """Check MERF compiance."""
        check_estimator(MERF())


if __name__ == "__main__":
    unittest.main()
