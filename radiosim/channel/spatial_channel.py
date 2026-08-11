"""
Spatial Multipath Channel Model for Direction of Arrival (DOA) Research.

This module provides a mathematical narrowband spatial propagation channel model
designed for Direction of Arrival (DOA) algorithm prototyping under multipath conditions.
"""

import numpy as np
from typing import List, Union, Optional


class UniformLinearArray:
	"""
	Uniform Linear Array (ULA) antenna model.
	
	Attributes:
		num_elements (int): Number of antenna elements (M).
		element_spacing (float): Spacing between adjacent elements in meters (d).
		carrier_frequency (float): Carrier frequency in Hz (fc).
		speed_of_light (float): Propagation speed of light in m/s (c).
	"""

	def __init__(
		self,
		num_elements: int,
		element_spacing: float,
		carrier_frequency: float,
		speed_of_light: float = 299792458.0
	):
		if num_elements < 1:
			raise ValueError("num_elements must be at least 1")
		if element_spacing <= 0:
			raise ValueError("element_spacing must be positive")
		if carrier_frequency <= 0:
			raise ValueError("carrier_frequency must be positive")

		self.num_elements = int(num_elements)
		self.element_spacing = float(element_spacing)
		self.carrier_frequency = float(carrier_frequency)
		self.speed_of_light = float(speed_of_light)

	@property
	def wavelength(self) -> float:
		"""Signal wavelength in meters (lambda = c / fc)."""
		return self.speed_of_light / self.carrier_frequency

	@property
	def element_positions(self) -> np.ndarray:
		"""Positions of array elements along 1D array axis in meters."""
		return np.arange(self.num_elements, dtype=np.float64) * self.element_spacing

	def steering_vector(self, angle: float, is_degrees: bool = True) -> np.ndarray:
		"""
		Calculates the array manifold / steering vector for a far-field narrowband signal arriving at theta.
		
		a_m(theta) = exp(-j * 2pi * m * d * sin(theta) / wavelength)
		
		Args:
			angle: Angle of arrival (theta).
			is_degrees: If True, angle is in degrees; if False, in radians.
			
		Returns:
			Complex array steering vector of shape (num_elements,).
		"""
		theta = np.radians(angle) if is_degrees else float(angle)
		m = np.arange(self.num_elements, dtype=np.float64)
		phase = -2.0 * np.pi * m * self.element_spacing * np.sin(theta) / self.wavelength
		return np.exp(1j * phase)


class SignalSource:
	"""
	Represents an RF signal source emitting a complex baseband waveform.
	
	Attributes:
		waveform (np.ndarray): Complex baseband signal s[n].
		angle_of_arrival (float): Direct line-of-sight angle of arrival (theta).
		is_degrees (bool): Whether angle_of_arrival is in degrees.
		power (float): Source transmission power / scale factor.
		identifier (str): Optional source identifier.
	"""

	def __init__(
		self,
		waveform: np.ndarray,
		angle_of_arrival: float = 0.0,
		is_degrees: bool = True,
		power: float = 1.0,
		identifier: Optional[str] = None
	):
		self.waveform = np.asarray(waveform, dtype=np.complex128)
		if self.waveform.ndim != 1:
			raise ValueError("waveform must be a 1D array")
		self.angle_of_arrival = float(angle_of_arrival)
		self.is_degrees = bool(is_degrees)
		self.power = float(power)
		self.identifier = identifier or "source_0"

	@property
	def aoa_rad(self) -> float:
		"""Angle of arrival in radians."""
		return np.radians(self.angle_of_arrival) if self.is_degrees else self.angle_of_arrival

	@property
	def aoa_deg(self) -> float:
		"""Angle of arrival in degrees."""
		return self.angle_of_arrival if self.is_degrees else np.degrees(self.angle_of_arrival)


class PropagationPath:
	"""
	Represents a single propagation path (direct or reflected).
	
	Attributes:
		angle_of_arrival (float): Angle of arrival for this path.
		is_degrees (bool): Whether angle_of_arrival is in degrees.
		complex_gain (complex): Complex path attenuation/gain a_k = |a_k| * exp(j * phase_offset).
		delay_samples (float): Path propagation delay in samples.
		delay_seconds (float): Path propagation delay in seconds.
		identifier (str): Optional path identifier.
	"""

	def __init__(
		self,
		angle_of_arrival: float = 0.0,
		is_degrees: bool = True,
		complex_gain: complex = 1.0 + 0j,
		relative_amplitude: Optional[float] = None,
		phase_offset: float = 0.0,
		delay_samples: float = 0.0,
		delay_seconds: float = 0.0,
		identifier: Optional[str] = None
	):
		self.angle_of_arrival = float(angle_of_arrival)
		self.is_degrees = bool(is_degrees)
		
		if relative_amplitude is not None:
			self.complex_gain = complex(relative_amplitude * np.exp(1j * phase_offset))
		else:
			self.complex_gain = complex(complex_gain)
			
		self.delay_samples = float(delay_samples)
		self.delay_seconds = float(delay_seconds)
		self.identifier = identifier or "path_0"

	@property
	def aoa_rad(self) -> float:
		"""Angle of arrival in radians."""
		return np.radians(self.angle_of_arrival) if self.is_degrees else self.angle_of_arrival

	@property
	def aoa_deg(self) -> float:
		"""Angle of arrival in degrees."""
		return self.angle_of_arrival if self.is_degrees else np.degrees(self.angle_of_arrival)


class SpatialMultipathChannel:
	"""
	Spatial Multipath Channel simulating per-element received signals across an antenna array.
	
	Given s[n] and K propagation paths, the received signal at antenna m is:
	x_m[n] = sum_k a_k * s[n - tau_k] * exp(-j * 2pi * fc * tau_k) * exp(-j * 2pi * m * d * sin(theta_k) / wavelength) + w_m[n]
	"""

	def __init__(
		self,
		antenna_array: UniformLinearArray,
		noise_power: float = 0.0,
		sample_rate: float = 1.0
	):
		self.antenna_array = antenna_array
		self.noise_power = float(noise_power)
		self.sample_rate = float(sample_rate)

	def simulate(
		self,
		sources: Union[SignalSource, List[SignalSource]],
		paths: Optional[Union[List[PropagationPath], List[List[PropagationPath]]]] = None,
		noise_power: Optional[float] = None
	) -> np.ndarray:
		"""
		Simulates spatial channel propagation and returns IQ matrix.
		
		Args:
			sources: A single SignalSource or list of SignalSources.
			paths: Propagation paths per source.
				- If None, each source gets 1 direct path using its AoA and gain=1.
				- If List[PropagationPath], applied to the single/first source.
				- If List[List[PropagationPath]], paths[i] corresponds to sources[i].
			noise_power: Override default noise power variance if provided.
			
		Returns:
			Complex IQ matrix of shape (num_antennas, num_samples).
		"""
		if isinstance(sources, SignalSource):
			source_list = [sources]
		else:
			source_list = list(sources)

		if len(source_list) == 0:
			raise ValueError("At least one SignalSource must be provided")

		num_samples = max(len(s.waveform) for s in source_list)
		num_antennas = self.antenna_array.num_elements

		# Normalize paths input into a list of path lists per source
		path_lists: List[List[PropagationPath]] = []
		if paths is None:
			for src in source_list:
				path_lists.append([
					PropagationPath(
						angle_of_arrival=src.angle_of_arrival,
						is_degrees=src.is_degrees,
						complex_gain=1.0 + 0j,
						delay_samples=0.0
					)
				])
		elif len(paths) > 0 and isinstance(paths[0], PropagationPath):
			# Single list of paths for the first source
			path_lists.append(paths)  # type: ignore
			for src in source_list[1:]:
				path_lists.append([
					PropagationPath(
						angle_of_arrival=src.angle_of_arrival,
						is_degrees=src.is_degrees,
						complex_gain=1.0 + 0j
					)
				])
		else:
			path_lists = paths  # type: ignore

		# Initialize complex IQ matrix for array: (M, N)
		received_iq = np.zeros((num_antennas, num_samples), dtype=np.complex128)

		for src_idx, src in enumerate(source_list):
			s_waveform = src.waveform
			s_power_scale = np.sqrt(src.power)
			src_paths = path_lists[src_idx] if src_idx < len(path_lists) else []

			for path in src_paths:
				# Total path gain including source power scaling
				path_gain = path.complex_gain * s_power_scale
				
				# Calculate delay in samples and seconds
				if path.delay_seconds > 0 and path.delay_samples == 0:
					delay_sec = path.delay_seconds
					delay_samp = path.delay_seconds * self.sample_rate
				else:
					delay_samp = path.delay_samples
					delay_sec = delay_samp / self.sample_rate if self.sample_rate > 0 else 0.0

				# Apply sample delay to baseband waveform
				delayed_signal = self._apply_delay(s_waveform, delay_samp, num_samples)

				# Carrier phase delay factor: exp(-j * 2pi * fc * tau_k)
				carrier_phase_factor = np.exp(-1j * 2.0 * np.pi * self.antenna_array.carrier_frequency * delay_sec)

				# Steering vector for path angle of arrival
				steering_vec = self.antenna_array.steering_vector(path.angle_of_arrival, is_degrees=path.is_degrees)

				# Outer product: steering_vec (M, 1) * delayed_signal (1, N)
				path_signal_matrix = (
					path_gain * carrier_phase_factor * np.outer(steering_vec, delayed_signal)
				)

				received_iq += path_signal_matrix

		# Add independent complex receiver AWGN
		effective_noise_power = noise_power if noise_power is not None else self.noise_power
		if effective_noise_power > 0:
			std_dev = np.sqrt(effective_noise_power / 2.0)
			noise = np.random.normal(0, std_dev, size=(num_antennas, num_samples)) + \
					1j * np.random.normal(0, std_dev, size=(num_antennas, num_samples))
			received_iq += noise

		return received_iq

	def _apply_delay(self, waveform: np.ndarray, delay_samples: float, target_length: int) -> np.ndarray:
		"""
		Applies time delay to waveform via discrete sample shift (zero-padded).
		"""
		out = np.zeros(target_length, dtype=np.complex128)
		int_delay = int(round(delay_samples))
		n_src = len(waveform)

		if int_delay >= 0:
			if int_delay < target_length:
				copy_len = min(n_src, target_length - int_delay)
				out[int_delay:int_delay + copy_len] = waveform[:copy_len]
		else:
			abs_delay = abs(int_delay)
			if abs_delay < n_src:
				copy_len = min(n_src - abs_delay, target_length)
				out[:copy_len] = waveform[abs_delay:abs_delay + copy_len]

		return out
