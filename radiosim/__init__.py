from .node import *
from .channel import (
	Channel as PythonChannel,
	UniformLinearArray,
	UniformCircularArray,
	SignalSource,
	PropagationPath,
	SpatialMultipathChannel,
	sample_covariance,
	bartlett_spectrum,
	music_spectrum,
	estimate_peaks
)

try:
	from .core import Channel
except (ImportError, RuntimeError):
	Channel = PythonChannel

