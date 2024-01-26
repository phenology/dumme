"""Collection of tests to verify input is parsed correctly.

Using fixtures in parametrized tests through request.getfixturevalue fixture:
https://engineeringfordatascience.com/posts/pytest_fixtures_with_parameterize/
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from dumme.dumme import MixedEffectsModel

from dumme.utils import DummeDataGenerator


@pytest.fixture
def pd_X():
    """Create a pandas dataframe for testing.

    Dataframe will look something like this:

             X_0       X_1       X_2    Z  cluster
    0   0.924864  0.845667  0.720978  1.0        0
    4  -0.605465 -0.222457 -1.221667  1.0        1
    5   1.214952 -0.792800  0.802716  1.0        1
    6   0.303006  1.329137 -1.041186  1.0        1

    """
    dg = DummeDataGenerator(m=0.6, sigma_b=4.5, sigma_e=1)
    X, _, _, _, _, _ = dg.generate_split_samples([1, 3], [3, 2], [1, 1])
    return X.drop("y", axis=1)


@pytest.fixture
def np_X(pd_X):
    return np.array(pd_X)


@pytest.fixture
def list_X(np_X):
    return np_X.tolist()


@pytest.fixture
def merf():
    return MixedEffectsModel(RandomForestRegressor(n_estimators=300, n_jobs=-1))


@pytest.mark.parametrize(
    "X,fit_kwargs",
    [
        # Valid: pandas dataframe with strings as column indices
        ("pd_X", {}),
        ("pd_X", dict(cluster_column="cluster")),
        ("pd_X", dict(cluster_column="cluster", random_effects=["Z"])),
        ("pd_X", dict(cluster_column="cluster", fixed_effects=["X_0", "X_1", "X_2"])),
        (
            "pd_X",
            dict(
                cluster_column="cluster",
                fixed_effects=["X_0", "X_1", "X_2"],
                random_effects=["Z"],
            ),
        ),
        ("pd_X", dict(fixed_effects=["X_0", "X_1", "X_2"], random_effects=["Z"])),
        ("pd_X", dict(fixed_effects=["X_0", "X_1", "X_2"])),
        ("pd_X", dict(random_effects=["Z"])),
        # Valid: numpy array with ints as column indices
        ("np_X", {}),
        ("np_X", dict(cluster_column=4)),
        ("np_X", dict(cluster_column=4, random_effects=[3])),
        ("np_X", dict(cluster_column=4, fixed_effects=[0, 1, 2])),
        ("np_X", dict(cluster_column=4, fixed_effects=[0, 1, 2], random_effects=[3])),
        ("np_X", dict(fixed_effects=[0, 1, 2], random_effects=[3])),
        ("np_X", dict(fixed_effects=[0, 1, 2])),
        ("np_X", dict(random_effects=[3])),
        # Valid: nested list with ints as column indices
        ("list_X", {}),
        ("list_X", dict(cluster_column=4)),
        ("list_X", dict(cluster_column=4, random_effects=[3])),
        ("list_X", dict(cluster_column=4, fixed_effects=[0, 1, 2])),
        ("list_X", dict(cluster_column=4, fixed_effects=[0, 1, 2], random_effects=[3])),
        ("list_X", dict(fixed_effects=[0, 1, 2], random_effects=[3])),
        ("list_X", dict(fixed_effects=[0, 1, 2])),
        ("list_X", dict(random_effects=[3])),
        # Valid: pandas array with ints as column indices
        ("pd_X", dict(cluster_column=4)),
        ("pd_X", dict(cluster_column=4, random_effects=[3])),
        ("pd_X", dict(cluster_column=4, fixed_effects=[0, 1, 2])),
        ("pd_X", dict(cluster_column=4, fixed_effects=[0, 1, 2], random_effects=[3])),
        ("pd_X", dict(fixed_effects=[0, 1, 2], random_effects=[3])),
        ("pd_X", dict(fixed_effects=[0, 1, 2])),
        ("pd_X", dict(random_effects=[3])),
        # Valid: pandas array with mixes type column indices
        ("pd_X", dict(cluster_column=4)),
        ("pd_X", dict(cluster_column=4, random_effects=["Z"])),
        ("pd_X", dict(cluster_column="cluster", fixed_effects=[0, 1, 2])),
        (
            "pd_X",
            dict(
                cluster_column=4,
                fixed_effects=["X_0", "X_1", "X_2"],
                random_effects=[3],
            ),
        ),
        ("pd_X", dict(fixed_effects=[0, 1, 2], random_effects=[3])),
        ("pd_X", dict(fixed_effects=["X_0", "X_1", "X_2"])),
        ("pd_X", dict(random_effects=["Z"])),
        # Valid: Z not as list
        ("pd_X", dict(random_effects="Z")),
        ("pd_X", dict(random_effects=3)),
        ("np_X", dict(random_effects=3)),
    ],
)
def test_valid_fit_input(X, fit_kwargs, merf, request):
    X = request.getfixturevalue(X)
    merf._parse_fit_kwargs(X, **fit_kwargs)

    expected_cluster_column = 4
    expected_random_effects = [3] if "random_effects" in fit_kwargs else []
    expected_fixed_effects = (
        [0, 1, 2]
        if "random_effects" in fit_kwargs or "fixed_effects" in fit_kwargs
        else [0, 1, 2, 3]
    )
    assert merf.cluster_column_ == expected_cluster_column
    assert merf.fixed_effects_ == expected_fixed_effects
    assert merf.random_effects_ == expected_random_effects


@pytest.mark.parametrize(
    "X,fit_kwargs",
    [
        # Invalid: numpy array with strings as column indices
        ("np_X", dict(cluster_column="cluster")),
        ("np_X", dict(cluster_column="cluster", random_effects=["Z"])),
        (
            "np_X",
            dict(
                cluster_column="cluster",
                fixed_effects=["X_0", "X_1", "X_2"],
            ),
        ),
        (
            "np_X",
            dict(
                cluster_column="cluster",
                fixed_effects=["X_0", "X_1", "X_2"],
                random_effects=["Z"],
            ),
        ),
        ("np_X", dict(fixed_effects=["X_0", "X_1", "X_2"], random_effects=["Z"])),
        ("np_X", dict(fixed_effects=["X_0", "X_1", "X_2"])),
        ("np_X", dict(random_effects=["Z"])),
        # Invalid: nested list with strings as column indices
        ("list_X", dict(cluster_column="cluster")),
        ("list_X", dict(cluster_column="cluster", random_effects=["Z"])),
        (
            "list_X",
            dict(
                cluster_column="cluster",
                fixed_effects=["X_0", "X_1", "X_2"],
            ),
        ),
        (
            "list_X",
            dict(
                cluster_column="cluster",
                fixed_effects=["X_0", "X_1", "X_2"],
                random_effects=["Z"],
            ),
        ),
        (
            "list_X",
            dict(
                fixed_effects=["X_0", "X_1", "X_2"],
                random_effects=["Z"],
            ),
        ),
        ("list_X", dict(fixed_effects=["X_0", "X_1", "X_2"])),
        ("list_X", dict(random_effects=["Z"])),
        # Invalid: non-str / non-int arguments / mixed lists
        ("X", dict(random_effects=2.4)),
        ("X", dict(random_effects=["a", 3, "b"])),
        ("np_X", dict(cluster_column=(1,))),
    ],
)
def test_invalid_fit_input(X, fit_kwargs, merf, request):
    with pytest.raises(ValueError):
        X = request.getfixturevalue(X)
        merf._parse_fit_kwargs(X, **fit_kwargs)
