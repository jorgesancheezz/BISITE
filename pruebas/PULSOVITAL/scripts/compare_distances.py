import numpy as np
from scipy.signal import welch

# Reference files
references = [
    'PULSOVITAL/results/003.npy',
    'PULSOVITAL/results/004.npy',
    'PULSOVITAL/results/005.npy',
    'PULSOVITAL/results/006.npy',
    'PULSOVITAL/results/007.npy'
]

# Load reference PSDs
ref_psds = []
for ref in references:
    data = np.load(ref)
    signal = data.mean(axis=(0, 2))
    _, psd = welch(signal, fs=250.0, nperseg=256, noverlap=128)
    ref_psds.append(psd)

# Load PSD for sintetico.npy
data_sintetico = np.load('PULSOVITAL/results/sintetico.npy')
signal_sintetico = data_sintetico.mean(axis=(0, 2))
_, psd_sintetico = welch(signal_sintetico, fs=250.0, nperseg=256, noverlap=128)

# Load PSD for synthetic_scale_0.1270_noise_0.05.npz
data_synthetic = np.load('PULSOVITAL/results/synthetic_scale_0.1270_noise_0.05.npz')['data']
signal_synthetic = data_synthetic.mean(axis=(0, 2))
_, psd_synthetic = welch(signal_synthetic, fs=250.0, nperseg=256, noverlap=128)

# Calculate average distances
dist_sintetico = np.mean([np.linalg.norm(psd_sintetico - ref_psd) for ref_psd in ref_psds])
dist_synthetic = np.mean([np.linalg.norm(psd_synthetic - ref_psd) for ref_psd in ref_psds])

print(f"Average distance to references:")
print(f"sintetico.npy: {dist_sintetico}")
print(f"synthetic_scale_0.1270_noise_0.05.npz: {dist_synthetic}")