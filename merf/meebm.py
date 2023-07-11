from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from merf.merf import MixedEffectsModel


class MEEBM(MixedEffectsModel):
    """Mixed-effects explainable boosting machines.

    Mixed-effects model with explainable boosting machine regressor as fixed
    effects model.
    """
    @property
    def fixed_effects_model(self):
        return ExplainableBoostingRegressor(**self.fe_kwargs)


class MERF(MixedEffectsModel):
    """Mixed-effects random forests.

    Mixed-effects model with random forest regressor as fixed effects model.
    """
    @property
    def fixed_effects_model(self):
        return RandomForestRegressor(**self.fe_kwargs)
