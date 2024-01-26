"""Tests that implementation adheres to sklearn api

https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""

from sklearn.tree import DecisionTreeRegressor
from dumme.dumme import MixedEffectsModel

from sklearn.utils.estimator_checks import parametrize_with_checks


@parametrize_with_checks(
    [
        DecisionTreeRegressor(),  # Ensure the checks work as expected
        MixedEffectsModel(),  # Check default dummy model
        MixedEffectsModel(DecisionTreeRegressor()),  # Combined ME model
        # MixedEffectsModel(RandomForestRegressor()),  # Original MERF fails due to of randomness
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
