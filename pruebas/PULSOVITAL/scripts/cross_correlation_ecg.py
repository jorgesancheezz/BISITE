import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

# Function to calculate cross-correlation
def calculate_cross_correlation(signal1, signal2):
    correlation = correlate(signal1, signal2, mode='full')
    lag = np.arange(-len(signal1) + 1, len(signal2))
    return lag, correlation

# Load reference ECG signals
reference_files = [
    'PULSOVITAL/results/003.npy',
    'PULSOVITAL/results/004.npy',
    'PULSOVITAL/results/005.npy',
    'PULSOVITAL/results/006.npy',
    'PULSOVITAL/results/007.npy'
]

# Load synthetic ECG signal
synthetic_file = 'PULSOVITAL/results/synthetic_scale_0.1290_noise_0.1.npz'
synthetic_data = np.load(synthetic_file)['data']
synthetic_signal = synthetic_data.mean(axis=(0, 2))

plt.figure(figsize=(12, 8))

# Calculate and plot cross-correlation for each reference file
for i, ref_file in enumerate(reference_files):
    ref_data = np.load(ref_file)
    ref_signal = ref_data.mean(axis=(0, 2))

    lag, correlation = calculate_cross_correlation(ref_signal, synthetic_signal)

    plt.subplot(len(reference_files), 1, i + 1)
    plt.plot(lag, correlation, label=f'Cross-correlation with {ref_file.split("/")[-1]}')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('PULSOVITAL/results/cross_correlation_ecg.png')
plt.close()

print("Cross-correlation analysis completed. Results saved to 'PULSOVITAL/results/cross_correlation_ecg.png'")