from PULSOVITAL.data.synthetic_loader import build_synthetic_loader, collect_samples
from scipy.signal import welch
import numpy as np
import sys

# Load reference PSD data
ref_data = np.load('PULSOVITAL/results/ref_psd_data.npz')
ref_psd = ref_data['ref_psd']

# Get scale value from command-line arguments
scale = float(sys.argv[1])

# Build synthetic loader and collect samples
loader = build_synthetic_loader(generator='wavelet', length=10000, alpha=0.9, noise=0.05, num_samples=128, scale=scale)
data = collect_samples(loader, max_samples=128)

# Calculate PSD difference
synth_freqs, synth_psd = welch(data[:, :, 0].mean(axis=0), fs=250.0, nperseg=256, noverlap=128)
diff = np.mean(np.abs(ref_psd - synth_psd))

# Save results
np.savez(f'PULSOVITAL/results/scale_{scale}_result.npz', scale=scale, diff=diff)