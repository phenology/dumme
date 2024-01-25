# DumME: Mixed Effects Dummy Model

This is an adaptation of https://github.com/manifoldai/merf. The most important changes are:

* This version is (mostly) compliant with the SKlearn API and can therefore be
  used with other frameworks such as
  [pycaret](https://pycaret.gitbook.io/docs/).
  * Fit API starts with X and y; Z and clusters are removed. Instead, optional kwargs to specify the columns of 'clusters', 'fixed_effects', and 'random_effects'.
  * Predict only accepts X as input. It is assumed new data is structured in the same way as the original training data.
* The main class was renamed to MixedEffectsModel (more general) with the
  scikit-learn dummy model as default.
* Package trimmed down to bare minimum but with modern package structure

We don't intend to develop this further. However, the changes could be ported to the original verison of the code. Notice https://github.com/manifoldai/merf/issues/68.

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
me_dummy.fit(X, y, cluster_column="cluster", fixed_effects=["X_0", "X_1", "X_2"], random_effects=["Z"])
```

To get the "original" MERF (but still with the new fit signature):

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, n_jobs=-1)
me_rf = MixedEffectsModel(rf)
me_rf.fit(X, y, cluster_column="cluster", fixed_effects=["X_0", "X_1", "X_2"], random_effects=["Z"])
```
