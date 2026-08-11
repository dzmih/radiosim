import unittest

import numpy as np

from radiosim.channel import (
	SignalSource,
	SpatialMultipathChannel,
	UniformLinearArray,
	bartlett_spectrum,
	estimate_peaks,
	music_spectrum,
	sample_covariance,
)


class TestDOA(unittest.TestCase):
	def setUp(self):
		self.fc = 1e9
		self.c = 3e8
		self.array = UniformLinearArray(
			num_elements=8,
			element_spacing=self.c / self.fc / 2,
			carrier_frequency=self.fc,
			speed_of_light=self.c,
		)
		self.angles = np.linspace(-90.0, 90.0, 1801)

	def test_covariance_is_hermitian(self):
		rng = np.random.default_rng(1)
		x = rng.normal(size=(8, 128)) + 1j * rng.normal(size=(8, 128))
		rxx = sample_covariance(x)
		self.assertEqual(rxx.shape, (8, 8))
		np.testing.assert_allclose(rxx, rxx.conj().T)

	def test_bartlett_finds_single_source(self):
		rng = np.random.default_rng(2)
		waveform = rng.normal(size=2048) + 1j * rng.normal(size=2048)
		source = SignalSource(waveform, angle_of_arrival=25.0)
		x = SpatialMultipathChannel(self.array, noise_power=0.01).simulate(source)
		angles, spectrum = bartlett_spectrum(x, self.array, self.angles)
		estimated = angles[np.argmax(spectrum)]
		self.assertAlmostEqual(estimated, 25.0, delta=0.2)

	def test_music_finds_two_sources(self):
		rng = np.random.default_rng(3)
		num_samples = 4096
		sources = [
			SignalSource(rng.normal(size=num_samples) + 1j * rng.normal(size=num_samples), -20.0),
			SignalSource(rng.normal(size=num_samples) + 1j * rng.normal(size=num_samples), 35.0),
		]
		x = SpatialMultipathChannel(self.array, noise_power=0.02).simulate(sources)
		angles, spectrum = music_spectrum(x, self.array, self.angles, num_sources=2)
		estimated = estimate_peaks(angles, spectrum, num_peaks=2, min_separation_deg=5.0)
		np.testing.assert_allclose(estimated, [-20.0, 35.0], atol=0.2)

	def test_invalid_source_count_is_rejected(self):
		x = np.ones((8, 32), dtype=np.complex128)
		with self.assertRaises(ValueError):
			music_spectrum(x, self.array, self.angles, num_sources=8)


if __name__ == "__main__":
	unittest.main()
