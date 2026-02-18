import numpy as np
from PULSOVITAL.data.synthetic_loader import build_synthetic_loader, collect_samples
from scipy.signal import welch

# Define parameters
alphas = [0.85, 0.9, 0.95]
scales = [round(x, 4) for x in np.arange(0.1150, 0.1351, 0.001)]
noise_levels = [0.03, 0.05, 0.07]
ref_file = 'PULSOVITAL/results/004.npy'

# Load reference data
ref_data = np.load(ref_file)
ref_freqs, ref_psd = welch(ref_data[:, :, 0].mean(axis=0), fs=250.0, nperseg=256, noverlap=128)

# Initialize results
results = []

# Calculate area between curves for each combination
for alpha in alphas:
    for scale in scales:
        for noise in noise_levels:
            # Generate synthetic data
            loader = build_synthetic_loader(
                generator='wavelet',
                length=10000,
                alpha=alpha,
                noise=noise,
                batch_size=4,
                num_samples=128,
                scale=scale
            )
            synth_data = collect_samples(loader, max_samples=128)
            synth_freqs, synth_psd = welch(synth_data[:, :, 0].mean(axis=0), fs=250.0, nperseg=256, noverlap=128)

            # Interpolate PSDs to ensure alignment
            common_freqs = np.intersect1d(ref_freqs, synth_freqs)
            ref_interp = np.interp(common_freqs, ref_freqs, ref_psd)
            synth_interp = np.interp(common_freqs, synth_freqs, synth_psd)

            # Calculate area between curves using numpy.trapz
            area = np.trapz(np.abs(ref_interp - synth_interp), common_freqs)
            results.append((alpha, scale, noise, area))

# Save results
with open('PULSOVITAL/results/area_between_curves.txt', 'w') as f:
    for alpha, scale, noise, area in results:
        f.write(f'Alpha: {alpha}, Scale: {scale}, Noise: {noise}, Area: {area}\n')

print("Area calculations complete. Results saved to PULSOVITAL/results/area_between_curves.txt")