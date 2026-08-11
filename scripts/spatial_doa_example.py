#!/usr/bin/env python3
"""
Direction of Arrival (DOA) Spatial Channel Demonstration Example.

This script demonstrates:
1. Creating a 5-element Uniform Linear Array (ULA).
2. Generating a complex baseband signal.
3. Adding direct and reflected multipath propagation paths.
4. Simulating the multi-antenna received IQ matrix.
5. Verifying unique inter-element phase signatures across antennas.
"""

import sys
import os

# Add package root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from radiosim.channel.spatial_channel import (
	UniformLinearArray,
	SignalSource,
	PropagationPath,
	SpatialMultipathChannel
)



def main():
	print("=" * 70)
	print("  RadioSim: Narrowband Spatial Multipath Channel & DOA Prototype  ")
	print("=" * 70)

	# 1. Setup Antenna Array: 5-element ULA at fc = 1.0 GHz with half-wavelength spacing
	fc = 1e9          # Carrier frequency: 1 GHz
	c = 299792458.0   # Speed of light in m/s
	wavelength = c / fc
	d = wavelength / 2.0  # Spacing = 0.15 meters

	num_antennas = 5
	array = UniformLinearArray(
		num_elements=num_antennas,
		element_spacing=d,
		carrier_frequency=fc,
		speed_of_light=c
	)

	print(f"\n[1] Antenna Array Configuration:")
	print(f"    - Elements:          {array.num_elements}")
	print(f"    - Carrier Freq (fc): {array.carrier_frequency / 1e6:.1f} MHz")
	print(f"    - Wavelength (lambda):{array.wavelength * 100:.2f} cm")
	print(f"    - Element Spacing:   {array.element_spacing * 100:.2f} cm (lambda/2)")
	print(f"    - Element Positions: {array.element_positions}")

	# 2. Generate Baseband Source Waveform (Sinusoid / Tone)
	num_samples = 100
	t = np.arange(num_samples)
	baseband_freq = 0.05  # Normalized digital frequency
	waveform = np.exp(1j * 2.0 * np.pi * baseband_freq * t)

	source = SignalSource(
		waveform=waveform,
		angle_of_arrival=30.0,
		is_degrees=True,
		power=1.0,
		identifier="Tx_Source_1"
	)

	print(f"\n[2] Signal Source:")
	print(f"    - ID:                {source.identifier}")
	print(f"    - Direct Line-of-Sight AoA: {source.aoa_deg:.1f} deg")
	print(f"    - Samples:           {len(source.waveform)}")

	# 3. Configure Multipath Propagation Paths
	paths = [
		# Direct Path (LoS)
		PropagationPath(
			angle_of_arrival=30.0,
			complex_gain=1.0 + 0j,
			delay_samples=0.0,
			identifier="Direct_Path_30deg"
		),
		# Reflected Path 1 (Ground / Wall reflection)
		PropagationPath(
			angle_of_arrival=-15.0,
			complex_gain=0.6 * np.exp(1j * np.pi / 3),  # Attenuated with phase shift
			delay_samples=3.0,
			identifier="Reflected_Path_m15deg"
		),
		# Reflected Path 2 (Building reflection)
		PropagationPath(
			angle_of_arrival=60.0,
			complex_gain=0.35 * np.exp(-1j * np.pi / 4),
			delay_samples=7.0,
			identifier="Reflected_Path_60deg"
		)
	]

	print(f"\n[3] Propagation Paths:")
	for p in paths:
		print(f"    - {p.identifier:22s} | AoA: {p.aoa_deg:6.1f} deg | Gain: {abs(p.complex_gain):.2f} | Delay: {p.delay_samples:.1f} samples")


	# 4. Simulate Channel & Generate Multi-Antenna Received IQ Matrix
	noise_power = 0.01  # Small AWGN variance
	channel = SpatialMultipathChannel(antenna_array=array, noise_power=noise_power)
	iq_matrix = channel.simulate(sources=source, paths=paths)

	print(f"\n[4] Received IQ Matrix:")
	print(f"    - Shape:             {iq_matrix.shape} (Antennas x Samples)")
	print(f"    - Data Type:         {iq_matrix.dtype}")

	# 5. Analyze and Print Per-Antenna Phase Progression (Sample n = 15)
	sample_idx = 15
	print(f"\n[5] Per-Antenna Spatial IQ Response at Sample n={sample_idx}:")
	print(f"    {'Antenna':<8} {'Position (m)':<15} {'Complex Value (I + jQ)':<30} {'Magnitude':<12} {'Phase (deg)':<12}")
	print("    " + "-" * 78)

	phases = []
	for m in range(num_antennas):
		val = iq_matrix[m, sample_idx]
		mag = np.abs(val)
		phase_deg = np.degrees(np.angle(val))
		phases.append(phase_deg)
		pos = array.element_positions[m]
		print(f"    Ant {m:<4}  {pos:<15.4f} {f'{val.real:.4f} + {val.imag:.4f}j':<30} {mag:<12.4f} {phase_deg:<12.2f}")

	# Calculate inter-element phase differences relative to element 0
	print(f"\n[6] Phase Shift Progression Relative to Antenna 0:")
	for m in range(num_antennas):
		diff = (phases[m] - phases[0] + 180) % 360 - 180
		print(f"    Antenna {m}: Delta Phase = {diff:+7.2f} deg")

	# 6. Optional Matplotlib Visualization
	try:
		import matplotlib.pyplot as plt
		
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
		
		# Plot I-channel waveforms for each antenna
		t_axis = np.arange(num_samples)
		for m in range(num_antennas):
			ax1.plot(t_axis[:40], iq_matrix[m, :40].real, label=f"Antenna {m}")
		ax1.set_title("Received In-Phase (I) Waveforms per Antenna (First 40 Samples)")
		ax1.set_xlabel("Sample Index")
		ax1.set_ylabel("Amplitude")
		ax1.grid(True, linestyle="--", alpha=0.6)
		ax1.legend(loc="upper right")

		# Plot spatial phase progression across antennas
		ax2.plot(range(num_antennas), phases, "o-", color="crimson", linewidth=2, markersize=8)
		ax2.set_title(f"Spatial Phase Profile Across Array (Sample n={sample_idx})")
		ax2.set_xlabel("Antenna Index (m)")
		ax2.set_ylabel("Phase (Degrees)")
		ax2.set_xticks(range(num_antennas))
		ax2.grid(True, linestyle="--", alpha=0.6)

		plt.tight_layout()
		plot_file = "spatial_doa_demo_plot.png"
		plt.savefig(plot_file, dpi=150)
		print(f"\n[Plot] Demo plot saved successfully to '{plot_file}'.")
	except ImportError:
		print("\n[Plot] matplotlib not installed; skipping plot generation.")

	print("\nDemonstration completed successfully.")


if __name__ == "__main__":
	main()
