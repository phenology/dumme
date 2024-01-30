[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10588922.svg)](https://doi.org/10.5281/zenodo.10588922)

# DumME: Mixed Effects Dummy Model

This is an adaptation of MERF (https://github.com/manifoldai/merf). The main
difference is that this version is fully compliant with the scikit-learn API.

Other difference include:

- The name: MERF was renamed to the more general MixedEffectsModel
- The default fixed-effects model: dummy model instead of random forest
- The package structure: stripped down to its core and then upgraded to use
  modern standards
- Test suite: using pytest instead of unittest

> [!CAUTION]
> We are currently not maintaining or developing this further. Ideally we would
> contribute our changes to the original version of MERF
> (see https://github.com/manifoldai/merf/issues/68).
> Do reach out if you want to build upon or collaborate with us on this.

## Using this version

Install via github:

```bash
pip install git+https://github.com/phenology/merf
```

Instantiate the dummy model:

```python
from dumme.dumme import MixedEffectsModel
from dumme.utils import DummeDataGenerator

# Get some sample data
dg = DummeDataGenerator(m=0.6, sigma_b=4.5, sigma_e=1)
df, _ = dg.generate_split_samples([1, 3], [3, 2], [1, 1])
y = df.pop("y")
x = df

# Fit a dummy model
# Notice the signature of the `fit` method: first X and y, and the other args are optional.
me_dummy = MixedEffectsModel()
me_dummy.fit(X, y)

# or
me_dummy.fit(X, y, cluster_column="cluster", fixed_effects=["X_0", "X_1", "X_2"], random_effects=["Z"])

# Predict only accepts X as input. It is assumed new data is structured
# in the same way as the original training data.
new_X = X.copy()
me_dummy.predict(new_X)
```

To get the "original" MERF (but still with the new fit signature):

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, n_jobs=-1)
me_rf = MixedEffectsModel(rf)
me_rf.fit(X, y)
```
