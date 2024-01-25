# DumME: Mixed Effects Dummy Model

This is an adaptation of https://github.com/manifoldai/merf. The most important changes are:

* This version is (mostly) [SKlearn
  compliant](https://scikit-learn.org/stable/developers/develop.html) and can
  therefore be used with other frameworks such as
  [pycaret](https://pycaret.gitbook.io/docs/).
  * Fit API starts with X and y; Z and clusters are removed. Instead, optional
    kwargs to specify the columns of 'clusters', 'fixed_effects', and
    'random_effects'.
  * Predict only accepts X as input. It is assumed new data is structured in the
    same way as the original training data.
* The main class was renamed to MixedEffectsModel (more general) with the
  scikit-learn dummy model as default.
* Package trimmed down to bare minimum but with modern package structure

> [!CAUTION]
> We don't intend to maintain or develop this further. However, we are happy to
> contribute our changes to the original version of MERF.
> Notice https://github.com/manifoldai/merf/issues/68.

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
me_dummy = MixedEffectsModel()
me_dummy.fit(X, y)  # This works now

# But you can still pass in additional arguments
me_dummy.fit(X, y, cluster_column="cluster", fixed_effects=["X_0", "X_1", "X_2"], random_effects=["Z"])
```

To get the "original" MERF (but still with the new fit signature):

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, n_jobs=-1)
me_rf = MixedEffectsModel(rf)
me_rf.fit(X, y)
```
