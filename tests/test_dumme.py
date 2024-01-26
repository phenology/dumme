import pickle

import numpy as np
from lightgbm import LGBMRegressor

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError

from dumme.utils import DummeDataGenerator
from dumme.viz import plot_merf_training_stats
from dumme.dumme import MixedEffectsModel


class Data:
    def __init__(self):
        np.random.seed(3187)

        dg = DummeDataGenerator(m=0.6, sigma_b=4.5, sigma_e=1)
        (
            train,
            test_known,
            test_new,
            _,
            _,
            _,
        ) = dg.generate_split_samples([1, 3], [3, 2], [1, 1])

        self.X_train = train
        self.y_train = train.pop("y")
        self.cluster_column = "cluster"
        self.fixed_effects = ["X_0", "X_1", "X_2"]
        self.random_effects = ["Z"]
        self.fit_kwargs = dict(
            cluster_column=self.cluster_column,
            fixed_effects=self.fixed_effects,
            random_effects=self.random_effects,
        )

        self.fit_kwargs_numpy = dict(
            cluster_column=4, fixed_effects=[0, 1, 2], random_effects=[3]
        )

        self.X_known = test_known
        self.y_known = test_known.pop("y")

        self.X_new = test_new
        self.y_new = test_new.pop("y")


@pytest.fixture
def data():
    return Data()


def test_not_fitted_error(data):
    m = MixedEffectsModel(RandomForestRegressor(n_estimators=300, n_jobs=-1))
    with pytest.raises(NotFittedError):
        m.predict(data.X_known)


def test_fit_and_predict_pandas(data):
    m = MixedEffectsModel(
        RandomForestRegressor(n_estimators=300, n_jobs=-1), max_iterations=5
    )

    # Train
    m.fit(data.X_train, data.y_train, **data.fit_kwargs)
    assert len(m.gll_history_) == 5
    assert len(m.val_loss_history_) == 0

    # Predict Known Clusters
    yhat_known = m.predict(data.X_known)
    assert len(yhat_known) == 5

    # Predict New Clusters
    yhat_new = m.predict(data.X_new)
    assert len(yhat_new) == 2


def test_fit_and_predict_numpy(data):
    m = MixedEffectsModel(
        RandomForestRegressor(n_estimators=300, n_jobs=-1), max_iterations=5
    )
    # Train
    m.fit(np.array(data.X_train), np.array(data.y_train), **data.fit_kwargs_numpy)
    assert len(m.val_loss_history_) == 0

    # Predict Known Clusters
    yhat_known = m.predict(np.array(data.X_known))
    assert len(yhat_known) == 5

    # Predict New Clusters
    yhat_new = m.predict(np.array(data.X_new))
    assert len(yhat_new) == 2


def test_early_stopping(data):
    np.random.seed(3187)

    # Create a MERF model with a high early stopping threshold
    m = MixedEffectsModel(
        RandomForestRegressor(n_estimators=300, n_jobs=-1),
        max_iterations=5,
        gll_early_stop_threshold=0.1,
    )

    # Fit
    m.fit(data.X_train, data.y_train, **data.fit_kwargs)

    # The number of iterations should be less than max_iterations
    assert len(m.gll_history_) < 5


def test_pickle(data):
    m = MixedEffectsModel(
        RandomForestRegressor(n_estimators=300, n_jobs=-1), max_iterations=5
    )
    # Train
    m.fit(data.X_train, data.y_train, **data.fit_kwargs)

    # Write to pickle file
    with open("model.pkl", "wb") as fin:
        pickle.dump(m, fin)

    # Read back from pickle file
    with open("model.pkl", "rb") as fout:
        m_pkl = pickle.load(fout)

    # Check that m is not the same object as m_pkl
    assert m_pkl != m

    # Predict Known Clusters
    yhat_known_pkl = m_pkl.predict(data.X_known)
    yhat_known = m.predict(data.X_known)
    np.testing.assert_almost_equal(yhat_known_pkl, yhat_known)

    # Predict New Clusters
    yhat_new_pkl = m_pkl.predict(data.X_new)
    yhat_new = m.predict(data.X_new)
    np.testing.assert_almost_equal(yhat_new_pkl, yhat_new)


def test_user_defined_fe_model(data):
    m = MixedEffectsModel(LGBMRegressor(), max_iterations=5)

    # Train
    m.fit(data.X_train, data.y_train, **data.fit_kwargs)
    assert len(m.gll_history_) == 5

    # Predict Known Clusters
    yhat_known = m.predict(data.X_known)
    assert len(yhat_known) == 5

    # Predict New Clusters
    yhat_new = m.predict(data.X_new)
    assert len(yhat_new) == 2


def test_validation(data):
    m = MixedEffectsModel(LGBMRegressor(), max_iterations=5)

    # Train
    m.fit(
        data.X_train,
        data.y_train,
        **data.fit_kwargs,
        X_val=data.X_known,
        y_val=data.y_known,
    )
    assert len(m.val_loss_history_) == 5

    # Predict Known Clusters
    yhat_known = m.predict(data.X_known)
    assert len(yhat_known) == 5

    # Predict New Clusters
    yhat_new = m.predict(data.X_new)
    assert len(yhat_new) == 2


def test_validation_numpy(data):
    m = MixedEffectsModel(
        RandomForestRegressor(n_estimators=300, n_jobs=-1), max_iterations=3
    )

    # Train
    m.fit(
        np.array(data.X_train),
        data.y_train,
        **data.fit_kwargs_numpy,
        X_val=np.array(data.X_known),
        y_val=np.array(data.y_known),
    )
    assert len(m.val_loss_history_) == 3

    # Predict Known Clusters
    yhat_known = m.predict(data.X_known)
    assert len(yhat_known) == 5

    # Predict New Clusters
    yhat_new = m.predict(data.X_new)
    assert len(yhat_new) == 2


def test_viz(data):
    m = MixedEffectsModel(LGBMRegressor(), max_iterations=5)

    # Train
    m.fit(data.X_train, data.y_train, **data.fit_kwargs)
    plot_merf_training_stats(m)
