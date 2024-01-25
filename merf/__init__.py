import logging

from .merf import MixedEffectsModel
from .utils import MERFDataGenerator
from .viz import plot_merf_training_stats

# TODO: remove this, not responsibility of a library to set logging behaviour.
# logging.basicConfig(format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)


# Version of the merf package
__version__ = "1.0"
