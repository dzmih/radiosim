import unittest
import numpy as np
from radiosim.channel.spatial_channel import (
	UniformLinearArray,
	SignalSource,
	PropagationPath,
	SpatialMultipathChannel
)


class TestSpatialMultipathChannel(unittest.TestCase):

	def setUp(self):
		self.fc = 1e9  # 1 GHz
		self.c = 3e8   # Speed of light
		self.wavelength = self.c / self.fc
		self.d = self.wavelength / 2.0  # Half-wavelength spacing
		self.num_antennas = 5
		self.array = UniformLinearArray(
			num_elements=self.num_antennas,
			element_spacing=self.d,
			carrier_frequency=self.fc,
			speed_of_light=self.c
		)

	def test_output_matrix_dimensions(self):
		"""Verify received IQ matrix shape is (num_antennas, num_samples)."""
		num_samples = 256
		waveform = np.ones(num_samples, dtype=np.complex128)
		source = SignalSource(waveform=waveform, angle_of_arrival=30.0)
		channel = SpatialMultipathChannel(antenna_array=self.array, noise_power=0.0)
		
		iq_matrix = channel.simulate(source)
		self.assertEqual(iq_matrix.shape, (self.num_antennas, num_samples))

	def test_identical_antennas_phase_progression(self):
		"""Verify element-to-element phase progression matches steering vector theory."""
		num_samples = 100
		aoa_deg = 30.0
		waveform = np.ones(num_samples, dtype=np.complex128)
		source = SignalSource(waveform=waveform, angle_of_arrival=aoa_deg)
		channel = SpatialMultipathChannel(antenna_array=self.array, noise_power=0.0)
		
		iq_matrix = channel.simulate(source)
		
		# Expected steering vector
		expected_sv = self.array.steering_vector(aoa_deg)
		
		# Pick first sample for each antenna
		rx_vector = iq_matrix[:, 0]
		
		# Normalize by first antenna element
		normalized_rx = rx_vector / rx_vector[0]
		normalized_sv = expected_sv / expected_sv[0]
		
		np.testing.assert_allclose(normalized_rx, normalized_sv, rtol=1e-5, atol=1e-5)

	def test_changing_source_angle_changes_phase_progression(self):
		"""Verify changing source angle produces corresponding steering vector phase changes."""
		waveform = np.ones(50, dtype=np.complex128)
		channel = SpatialMultipathChannel(antenna_array=self.array, noise_power=0.0)
		
		source1 = SignalSource(waveform=waveform, angle_of_arrival=15.0)
		source2 = SignalSource(waveform=waveform, angle_of_arrival=45.0)
		
		iq1 = channel.simulate(source1)
		iq2 = channel.simulate(source2)
		
		# Phase differences across element 0 and 1
		phase_diff1 = np.angle(iq1[1, 0] * np.conj(iq1[0, 0]))
		phase_diff2 = np.angle(iq2[1, 0] * np.conj(iq2[0, 0]))
		
		# Theoretical expected phase differences
		expected_phase_diff1 = -2.0 * np.pi * self.d * np.sin(np.radians(15.0)) / self.wavelength
		expected_phase_diff2 = -2.0 * np.pi * self.d * np.sin(np.radians(45.0)) / self.wavelength
		
		self.assertNotAlmostEqual(phase_diff1, phase_diff2, places=3)
		self.assertAlmostEqual(phase_diff1, expected_phase_diff1, places=5)
		self.assertAlmostEqual(phase_diff2, expected_phase_diff2, places=5)

	def test_changing_path_delay_changes_waveform_alignment(self):
		"""Verify path delay shifts the received waveform alignment in time."""
		num_samples = 20
		waveform = np.zeros(num_samples, dtype=np.complex128)
		waveform[0] = 1.0 + 0j  # Impulse at n=0
		
		source = SignalSource(waveform=waveform, angle_of_arrival=0.0)
		delay_samples = 4
		path = PropagationPath(angle_of_arrival=0.0, delay_samples=delay_samples)
		
		channel = SpatialMultipathChannel(antenna_array=self.array, noise_power=0.0)
		iq_matrix = channel.simulate(source, paths=[path])
		
		# At theta=0, all elements receive impulse at sample index = delay_samples
		self.assertEqual(np.abs(iq_matrix[0, delay_samples]), 1.0)
		self.assertEqual(np.abs(iq_matrix[0, 0]), 0.0)
		self.assertEqual(np.abs(iq_matrix[3, delay_samples]), 1.0)

	def test_zero_noise_deterministic_behavior(self):
		"""Verify zero-noise simulation returns deterministic identical matrices."""
		waveform = np.random.randn(100) + 1j * np.random.randn(100)
		source = SignalSource(waveform=waveform, angle_of_arrival=20.0)
		channel = SpatialMultipathChannel(antenna_array=self.array, noise_power=0.0)
		
		iq1 = channel.simulate(source)
		iq2 = channel.simulate(source)
		
		np.testing.assert_array_equal(iq1, iq2)

	def test_independent_noise_between_antenna_channels(self):
		"""Verify noise is independent across antenna channels."""
		num_samples = 10000
		waveform = np.zeros(num_samples, dtype=np.complex128)
		source = SignalSource(waveform=waveform, angle_of_arrival=0.0)
		noise_power = 2.0
		
		channel = SpatialMultipathChannel(antenna_array=self.array, noise_power=noise_power)
		iq_matrix = channel.simulate(source, noise_power=noise_power)
		
		# Verify variance per antenna is close to noise_power
		for m in range(self.num_antennas):
			antenna_noise = iq_matrix[m, :]
			measured_var = np.var(antenna_noise)
			self.assertAlmostEqual(measured_var, noise_power, delta=0.2)
			
		# Verify cross-correlation between antenna 0 and 1 is small
		corr = np.abs(np.corrcoef(iq_matrix[0, :], iq_matrix[1, :])[0, 1])
		self.assertLess(corr, 0.05)

	def test_one_direct_plus_one_reflected_path_sum(self):
		"""Verify 1 direct path + 1 reflected path produces expected linear combination sum."""
		waveform = np.ones(50, dtype=np.complex128)
		source = SignalSource(waveform=waveform)
		
		direct_path = PropagationPath(angle_of_arrival=30.0, complex_gain=1.0 + 0j, delay_samples=0.0)
		reflected_path = PropagationPath(angle_of_arrival=-20.0, complex_gain=0.5 * np.exp(1j * np.pi / 4), delay_samples=2.0)
		
		channel = SpatialMultipathChannel(antenna_array=self.array, noise_power=0.0)
		
		# Individual path outputs
		iq_direct = channel.simulate(source, paths=[direct_path])
		iq_reflected = channel.simulate(source, paths=[reflected_path])
		
		# Combined multipath output
		iq_combined = channel.simulate(source, paths=[direct_path, reflected_path])
		
		expected_sum = iq_direct + iq_reflected
		np.testing.assert_allclose(iq_combined, expected_sum, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
	unittest.main()
