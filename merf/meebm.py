from interpret.glassbox import ExplainableBoostingRegressor
from merf.merf import MERF


def builder(**kwargs):
    return ExplainableBoostingRegressor(**kwargs)


class MEEBM(MERF):
    def __init__(
        self,
        gll_early_stop_threshold=None,
        max_iterations=20,
        **fe_kwargs,
    ):
        super().__init__(
            fixed_effects_model=builder,
            max_iterations=max_iterations,
            gll_early_stop_threshold=gll_early_stop_threshold,
            **fe_kwargs,
        )
