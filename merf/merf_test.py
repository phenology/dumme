"""
MERF Unit Tests

Run with this command for verbose output:
> python tests.py -v

:copyright: 2017 Manifold, Inc.
:author: Sourav Dey <sdey@manifold.ai>
"""
import pickle
import unittest

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from merf import MERF
from utils import MERFDataGenerator
from viz import plot_merf_training_stats


class DataGenerationTest(unittest.TestCase):
    def test_create_cluster_sizes(self):
        clusters = MERFDataGenerator.create_cluster_sizes_array([1, 2, 3], 1)
        self.assertListEqual(clusters, [1, 2, 3])

        clusters = MERFDataGenerator.create_cluster_sizes_array([30, 20, 7], 3)
        self.assertListEqual(clusters, [30, 30, 30, 20, 20, 20, 7, 7, 7])

    def test_generate_samples(self):
        dg = MERFDataGenerator(m=0.6, sigma_b=4.5, sigma_e=1)
        df, ptev, prev = dg.generate_samples([1, 2, 3])
        # check columns
        self.assertListEqual(df.columns.tolist(), ["y", "X_0", "X_1", "X_2", "Z", "cluster"])
        # check length
        self.assertEqual(len(df), 6)
        # check cluster sizes
        self.assertEqual(len(df[df["cluster"] == 0]), 1)
        self.assertEqual(len(df[df["cluster"] == 1]), 2)
        self.assertEqual(len(df[df["cluster"] == 2]), 3)

    def test_generate_split_samples(self):
        dg = MERFDataGenerator(m=0.7, sigma_b=2.7, sigma_e=1)
        train, test_known, test_new, training_ids, ptev, prev = dg.generate_split_samples([1, 3], [3, 2], [1, 1])
        # check all have same columns
        self.assertListEqual(train.columns.tolist(), ["y", "X_0", "X_1", "X_2", "Z", "cluster"])
        self.assertListEqual(test_known.columns.tolist(), ["y", "X_0", "X_1", "X_2", "Z", "cluster"])
        self.assertListEqual(test_new.columns.tolist(), ["y", "X_0", "X_1", "X_2", "Z", "cluster"])

        # check length
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test_known), 5)
        self.assertEqual(len(test_new), 2)

        # check cluster sizes
        self.assertEqual(len(train[train["cluster"] == 0]), 1)
        self.assertEqual(len(train[train["cluster"] == 1]), 3)
        self.assertEqual(len(test_known[test_known["cluster"] == 0]), 3)
        self.assertEqual(len(test_known[test_known["cluster"] == 1]), 2)
        self.assertEqual(len(test_new[test_new["cluster"] == 2]), 1)
        self.assertEqual(len(test_new[test_new["cluster"] == 3]), 1)

        # Check training ids
        self.assertListEqual(training_ids.tolist(), [0, 1])

    def test_ohe_clusters(self):
        training_cluster_ids = np.array([0, 1, 2, 3])
        # Training like encoding -- all categories in matrix
        X_ohe = MERFDataGenerator.ohe_clusters(
            pd.Series([0, 0, 1, 2, 2, 2, 3]), training_cluster_ids=training_cluster_ids
        )
        # check columns and sums
        self.assertListEqual(X_ohe.columns.tolist(), ["cluster_0", "cluster_1", "cluster_2", "cluster_3"])
        self.assertListEqual(X_ohe.sum().tolist(), [2, 1, 3, 1])

        # New encoding -- no categories in matrix
        X_ohe = MERFDataGenerator.ohe_clusters(pd.Series([4, 4, 5, 6, 6, 7]), training_cluster_ids=training_cluster_ids)
        # check columns and sums
        self.assertListEqual(X_ohe.columns.tolist(), ["cluster_0", "cluster_1", "cluster_2", "cluster_3"])
        self.assertListEqual(X_ohe.sum().tolist(), [0, 0, 0, 0])

        # Mixed encoding -- some categories in matrix
        X_ohe = MERFDataGenerator.ohe_clusters(
            pd.Series([1, 1, 3, 0, 0, 4, 5, 6, 6, 7]), training_cluster_ids=training_cluster_ids
        )
        # check columns and sums
        self.assertListEqual(X_ohe.columns.tolist(), ["cluster_0", "cluster_1", "cluster_2", "cluster_3"])
        self.assertListEqual(X_ohe.sum().tolist(), [2, 2, 0, 1])


class MERFTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(3187)

        dg = MERFDataGenerator(m=0.6, sigma_b=4.5, sigma_e=1)
        train, test_known, test_new, train_cluster_ids, ptev, prev = dg.generate_split_samples([1, 3], [3, 2], [1, 1])

        self.X_train = train
        self.y_train = train.pop("y")
        self.cluster_column = "cluster"
        self.fixed_effects = ["X_0", "X_1", "X_2"]
        self.random_effects = ["Z"]
        self.fit_kwargs = dict(
            cluster_column=self.cluster_column, fixed_effects=self.fixed_effects, random_effects=self.random_effects
        )

        self.fit_kwargs_numpy = dict(cluster_column=4, fixed_effects=[0, 1, 2], random_effects=[3])

        self.X_known = test_known
        self.y_known = test_known.pop("y")

        self.X_new = test_new
        self.y_new = test_new.pop("y")

    def test_not_fitted_error(self):
        m = MERF()
        with self.assertRaises(NotFittedError):
            m.predict(self.X_known)

    def test_fit_and_predict_pandas(self):
        m = MERF(max_iterations=5)
        # Train
        m.fit(self.X_train, self.y_train, **self.fit_kwargs)
        self.assertEqual(len(m.gll_history_), 5)
        self.assertEqual(len(m.val_loss_history_), 0)
        # Predict Known Clusters
        yhat_known = m.predict(self.X_known)
        self.assertEqual(len(yhat_known), 5)
        # Predict New Clusters
        yhat_new = m.predict(self.X_new)
        self.assertEqual(len(yhat_new), 2)

    def test_fit_and_predict_numpy(self):
        m = MERF(max_iterations=5)
        # Train
        m.fit(np.array(self.X_train), np.array(self.y_train), **self.fit_kwargs_numpy)
        self.assertEqual(len(m.val_loss_history_), 0)
        # Predict Known Clusters
        yhat_known = m.predict(np.array(self.X_known))
        self.assertEqual(len(yhat_known), 5)
        # Predict New Clusters
        yhat_new = m.predict(np.array(self.X_new))
        self.assertEqual(len(yhat_new), 2)

    def test_early_stopping(self):
        np.random.seed(3187)
        # Create a MERF model with a high early stopping threshold
        m = MERF(max_iterations=5, gll_early_stop_threshold=0.1)
        # Fit
        m.fit(self.X_train, self.y_train, **self.fit_kwargs)
        # The number of iterations should be less than max_iterations
        self.assertTrue(len(m.gll_history_) < 5)

    def test_pickle(self):
        m = MERF(max_iterations=5)
        # Train
        m.fit(self.X_train, self.y_train, **self.fit_kwargs)

        # Write to pickle file
        with open("model.pkl", "wb") as fin:
            pickle.dump(m, fin)

        # Read back from pickle file
        with open("model.pkl", "rb") as fout:
            m_pkl = pickle.load(fout)

        # Check that m is not the same object as m_pkl
        self.assertIsNot(m_pkl, m)
        # Predict Known Clusters
        yhat_known_pkl = m_pkl.predict(self.X_known)
        yhat_known = m.predict(self.X_known)
        assert_almost_equal(yhat_known_pkl, yhat_known)
        # Predict New Clusters
        yhat_new_pkl = m_pkl.predict(self.X_new)
        yhat_new = m.predict(self.X_new)
        assert_almost_equal(yhat_new_pkl, yhat_new)

    def test_user_defined_fe_model(self):
        lgbm = LGBMRegressor
        m = MERF(fixed_effects_model=lgbm, max_iterations=5)
        # Train
        m.fit(self.X_train, self.y_train, **self.fit_kwargs)
        self.assertEqual(len(m.gll_history_), 5)
        # Predict Known Clusters
        yhat_known = m.predict(self.X_known)
        self.assertEqual(len(yhat_known), 5)
        # Predict New Clusters
        yhat_new = m.predict(self.X_new)
        self.assertEqual(len(yhat_new), 2)

    def test_validation(self):
        lgbm = LGBMRegressor
        m = MERF(fixed_effects_model=lgbm, max_iterations=5)
        # Train
        m.fit(self.X_train, self.y_train, **self.fit_kwargs, X_val=self.X_known, y_val=self.y_known)
        self.assertEqual(len(m.val_loss_history_), 5)
        # Predict Known Clusters
        yhat_known = m.predict(self.X_known)
        self.assertEqual(len(yhat_known), 5)
        # Predict New Clusters
        yhat_new = m.predict(self.X_new)
        self.assertEqual(len(yhat_new), 2)

    def test_validation_numpy(self):
        m = MERF(max_iterations=3)
        # Train
        m.fit(
            np.array(self.X_train),
            self.y_train,
            **self.fit_kwargs_numpy,
            X_val=np.array(self.X_known),
            y_val=np.array(self.y_known),
        )
        self.assertEqual(len(m.val_loss_history_), 3)
        # Predict Known Clusters
        yhat_known = m.predict(self.X_known)
        self.assertEqual(len(yhat_known), 5)
        # Predict New Clusters
        yhat_new = m.predict(self.X_new)
        self.assertEqual(len(yhat_new), 2)

    def test_viz(self):
        lgbm = LGBMRegressor
        m = MERF(fixed_effects_model=lgbm, max_iterations=5)
        # Train
        m.fit(self.X_train, self.y_train, **self.fit_kwargs)
        plot_merf_training_stats(m)


class MerfInputTests(unittest.TestCase):
    """Collection of tests to verify input is parsed correctly."""

    def setUp(self):
        """Create a pandas dataframe for testing.

        Dataframe will look something like this:

                 X_0       X_1       X_2    Z  cluster
        0   0.924864  0.845667  0.720978  1.0        0
        4  -0.605465 -0.222457 -1.221667  1.0        1
        5   1.214952 -0.792800  0.802716  1.0        1
        6   0.303006  1.329137 -1.041186  1.0        1

        """
        dg = MERFDataGenerator(m=0.6, sigma_b=4.5, sigma_e=1)
        X, _, _, _, _, _ = dg.generate_split_samples([1, 3], [3, 2], [1, 1])
        self.X = X.drop("y", axis=1)

    def assert_valid_fit_input(self, X, *args, **kwargs):
        str_args = ", ".join(["X"] + list(args) + [f"{k}:{v}" for k, v in kwargs.items()])
        print(f"Test call: _parse_fit_input({str_args})")

        merf = MERF()
        merf._parse_fit_kwargs(X, *args, **kwargs)

        expected_cluster_column = 4
        expected_random_effects = [3] if "random_effects" in kwargs else []
        expected_fixed_effects = [0, 1, 2] if "random_effects" in kwargs or "fixed_effects" in kwargs else [0, 1, 2, 3]

        assert merf.cluster_column_ == expected_cluster_column
        assert merf.fixed_effects_ == expected_fixed_effects
        assert merf.random_effects_ == expected_random_effects

    def assert_invalid_fit_input(self, X, *args, **kwargs):
        str_args = ", ".join(["X"] + list(args) + [f"{k}:{v}" for k, v in kwargs.items()])
        print(f"Test call: _parse_fit_input({str_args})")

        merf = MERF()

        with self.assertRaises(ValueError):
            merf._parse_fit_kwargs(X, *args, **kwargs)

    def test_parse_fit_input(self):
        X = self.X

        # Valid: pandas dataframe with strings as column indices
        self.assert_valid_fit_input(X)
        self.assert_valid_fit_input(X, cluster_column="cluster")
        self.assert_valid_fit_input(X, cluster_column="cluster", random_effects=["Z"])
        self.assert_valid_fit_input(X, cluster_column="cluster", fixed_effects=["X_0", "X_1", "X_2"])
        self.assert_valid_fit_input(
            X, cluster_column="cluster", fixed_effects=["X_0", "X_1", "X_2"], random_effects=["Z"]
        )
        self.assert_valid_fit_input(X, fixed_effects=["X_0", "X_1", "X_2"], random_effects=["Z"])
        self.assert_valid_fit_input(X, fixed_effects=["X_0", "X_1", "X_2"])
        self.assert_valid_fit_input(X, random_effects=["Z"])

        # Valid: numpy array with ints as column indices
        self.assert_valid_fit_input(np.array(X))
        self.assert_valid_fit_input(np.array(X), cluster_column=4)
        self.assert_valid_fit_input(np.array(X), cluster_column=4, random_effects=[3])
        self.assert_valid_fit_input(np.array(X), cluster_column=4, fixed_effects=[0, 1, 2])
        self.assert_valid_fit_input(np.array(X), cluster_column=4, fixed_effects=[0, 1, 2], random_effects=[3])
        self.assert_valid_fit_input(np.array(X), fixed_effects=[0, 1, 2], random_effects=[3])
        self.assert_valid_fit_input(np.array(X), fixed_effects=[0, 1, 2])
        self.assert_valid_fit_input(np.array(X), random_effects=[3])

        # Valid: pandas array with ints as column indices
        self.assert_valid_fit_input(X)
        self.assert_valid_fit_input(X, cluster_column=4)
        self.assert_valid_fit_input(X, cluster_column=4, random_effects=[3])
        self.assert_valid_fit_input(X, cluster_column=4, fixed_effects=[0, 1, 2])
        self.assert_valid_fit_input(X, cluster_column=4, fixed_effects=[0, 1, 2], random_effects=[3])
        self.assert_valid_fit_input(X, fixed_effects=[0, 1, 2], random_effects=[3])
        self.assert_valid_fit_input(X, fixed_effects=[0, 1, 2])
        self.assert_valid_fit_input(X, random_effects=[3])

        # Valid: pandas array with mixes type column indices
        self.assert_valid_fit_input(X)
        self.assert_valid_fit_input(X, cluster_column=4)
        self.assert_valid_fit_input(X, cluster_column=4, random_effects=["Z"])
        self.assert_valid_fit_input(X, cluster_column="cluster", fixed_effects=[0, 1, 2])
        self.assert_valid_fit_input(X, cluster_column=4, fixed_effects=["X_0", "X_1", "X_2"], random_effects=[3])
        self.assert_valid_fit_input(X, fixed_effects=[0, 1, 2], random_effects=[3])
        self.assert_valid_fit_input(X, fixed_effects=["X_0", "X_1", "X_2"])
        self.assert_valid_fit_input(X, random_effects=["Z"])

        # Valid: Z not as list
        self.assert_valid_fit_input(X, random_effects="Z")
        self.assert_valid_fit_input(X, random_effects=3)
        self.assert_valid_fit_input(np.array(X), random_effects=3)

        # Invalid: numpy array with strings as column indices
        self.assert_invalid_fit_input(np.array(X), cluster_column="cluster")
        self.assert_invalid_fit_input(np.array(X), cluster_column="cluster", random_effects=["Z"])
        self.assert_invalid_fit_input(np.array(X), cluster_column="cluster", fixed_effects=["X_0", "X_1", "X_2"])
        self.assert_invalid_fit_input(
            np.array(X), cluster_column="cluster", fixed_effects=["X_0", "X_1", "X_2"], random_effects=["Z"]
        )
        self.assert_invalid_fit_input(np.array(X), fixed_effects=["X_0", "X_1", "X_2"], random_effects=["Z"])
        self.assert_invalid_fit_input(np.array(X), fixed_effects=["X_0", "X_1", "X_2"])
        self.assert_invalid_fit_input(np.array(X), random_effects=["Z"])

        # Invalid: non-str / non-int arguments / mixed lists
        self.assert_invalid_fit_input(X, random_effects=2.4)
        self.assert_invalid_fit_input(X, random_effects=["a", 3, "b"])
        self.assert_invalid_fit_input(np.array(X), cluster_column=(1,))


if __name__ == "__main__":
    unittest.main()
