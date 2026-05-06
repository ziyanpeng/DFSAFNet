import warnings

from . import train
from . import losses
from . import metrics
from . import metrics2
warnings.warn(
    "`smp.utils` module is deprecated and will be removed in future releases.",
    DeprecationWarning,
)


