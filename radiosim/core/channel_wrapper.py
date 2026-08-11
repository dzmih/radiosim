import ctypes
import os

try:
	_lib = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + "/build/lib/libchannel.so")
	_lib.channel_new.argtypes = (ctypes.c_float,)
except (FileNotFoundError, OSError):
	_lib = None

class Channel:
	def __init__(self, noise_energy):
		if _lib is None:
			raise RuntimeError("C++ Channel library (libchannel.so) is not compiled.")
		self.obj = _lib.channel_new(noise_energy)

	def add_node(self, txport, rxport, buffer_size=512):
		if _lib is None:
			raise RuntimeError("C++ Channel library (libchannel.so) is not compiled.")
		_lib.channel_add_node(self.obj, txport, rxport, buffer_size)

	def start(self):
		if _lib is None:
			raise RuntimeError("C++ Channel library (libchannel.so) is not compiled.")
		_lib.channel_start(self.obj)

