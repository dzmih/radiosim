"""Direction-of-arrival estimators for spatial IQ snapshots.

The functions in this module operate on an IQ matrix with shape ``(M, N)``:
``M`` antenna elements and ``N`` simultaneous time samples.
"""

from typing import Tuple

import numpy as np

from .spatial_channel import UniformLinearArray


def _validate_iq(iq: np.ndarray) -> np.ndarray:
	"""Return IQ data as a non-empty two-dimensional complex array."""
	x = np.asarray(iq, dtype=np.complex128)
	if x.ndim != 2:
		raise ValueError("iq must have shape (num_antennas, num_samples)")
	if 0 in x.shape:
		raise ValueError("iq must contain at least one antenna and one sample")
	return x


def _scan_angles(angles: np.ndarray) -> np.ndarray:
	grid = np.asarray(angles, dtype=float)
	if grid.ndim != 1 or grid.size == 0:
		raise ValueError("angles must be a non-empty one-dimensional array")
	return grid


def _steering_matrix(array: UniformLinearArray, angles: np.ndarray) -> np.ndarray:
	return np.column_stack([array.steering_vector(angle) for angle in angles])


def sample_covariance(iq: np.ndarray) -> np.ndarray:
	"""Estimate the spatial covariance matrix ``R = X X^H / N``."""
	x = _validate_iq(iq)
	return (x @ x.conj().T) / x.shape[1]


def bartlett_spectrum(
	iq: np.ndarray,
	array: UniformLinearArray,
	angles: np.ndarray,
	normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute the conventional Bartlett spatial spectrum."""
	x = _validate_iq(iq)
	grid = _scan_angles(angles)
	if x.shape[0] != array.num_elements:
		raise ValueError("iq antenna count must match array.num_elements")

	rxx = sample_covariance(x)
	steering = _steering_matrix(array, grid)
	power = np.real(np.sum(steering.conj() * (rxx @ steering), axis=0))
	power = np.maximum(power, 0.0)
	if normalize and np.max(power) > 0:
		power = power / np.max(power)
	return grid, power


def music_spectrum(
	iq: np.ndarray,
	array: UniformLinearArray,
	angles: np.ndarray,
	num_sources: int,
	normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute the MUSIC pseudospectrum.

	``num_sources`` is the expected number of independent signal sources and
	must be smaller than the number of antenna elements.
	"""
	x = _validate_iq(iq)
	grid = _scan_angles(angles)
	if x.shape[0] != array.num_elements:
		raise ValueError("iq antenna count must match array.num_elements")
	if not 0 < int(num_sources) < array.num_elements:
		raise ValueError("num_sources must be between 1 and num_antennas - 1")

	rxx = sample_covariance(x)
	_, eigenvectors = np.linalg.eigh(rxx)
	noise_subspace = eigenvectors[:, :array.num_elements - int(num_sources)]
	steering = _steering_matrix(array, grid)
	projection = noise_subspace.conj().T @ steering
	denominator = np.sum(np.abs(projection) ** 2, axis=0)
	spectrum = 1.0 / np.maximum(denominator, np.finfo(float).tiny)
	if normalize and np.max(spectrum) > 0:
		spectrum = spectrum / np.max(spectrum)
	return grid, spectrum


def estimate_peaks(
	angles: np.ndarray,
	spectrum: np.ndarray,
	num_peaks: int = 1,
	min_separation_deg: float = 1.0,
) -> np.ndarray:
	"""Return the strongest separated local maxima in ascending angle order."""
	grid = _scan_angles(angles)
	values = np.asarray(spectrum, dtype=float)
	if values.shape != grid.shape:
		raise ValueError("angles and spectrum must have the same shape")
	if num_peaks < 1 or min_separation_deg < 0:
		raise ValueError("num_peaks must be positive and separation non-negative")

	if values.size == 1:
		candidates = np.array([0])
	else:
		local_max = np.ones(values.size, dtype=bool)
		local_max[1:-1] = (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
		candidates = np.flatnonzero(local_max)
	if candidates.size == 0:
		candidates = np.array([int(np.argmax(values))])

	selected = []
	for index in candidates[np.argsort(values[candidates])[::-1]]:
		if all(abs(grid[index] - grid[chosen]) >= min_separation_deg for chosen in selected):
			selected.append(int(index))
			if len(selected) == num_peaks:
				break
	return np.sort(grid[selected])


__all__ = [
	"sample_covariance",
	"bartlett_spectrum",
	"music_spectrum",
	"estimate_peaks",
]
