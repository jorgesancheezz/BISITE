import numpy as np
from scipy.signal import correlate
import glob

# Function to calculate cross-correlation
def calculate_cross_correlation(signal1, signal2):
    correlation = correlate(signal1, signal2, mode='full')
    return np.max(correlation)  # Return the maximum correlation value

# Load reference ECG signals
reference_files = [
    'PULSOVITAL/results/003.npy',
    'PULSOVITAL/results/004.npy',
    'PULSOVITAL/results/005.npy',
    'PULSOVITAL/results/006.npy',
    'PULSOVITAL/results/007.npy'
]

# Load all synthetic ECG files with scales from 0.0 to 0.10
synthetic_files = [
    f'PULSOVITAL/results/synthetic_scale_{scale:.2f}_noise_0.01.npz' for scale in np.arange(0.0, 0.11, 0.01)
]

# Dictionary to store average correlations
average_correlations = {}

# Iterate over synthetic files
for synthetic_file in synthetic_files:
    synthetic_data = np.load(synthetic_file)['data']
    synthetic_signal = synthetic_data.mean(axis=(0, 2))

    # Calculate correlation with each reference file
    correlations = []
    for ref_file in reference_files:
        ref_data = np.load(ref_file)
        ref_signal = ref_data.mean(axis=(0, 2))

        max_correlation = calculate_cross_correlation(ref_signal, synthetic_signal)
        correlations.append(max_correlation)

    # Calculate average correlation for the synthetic file
    average_correlations[synthetic_file] = np.mean(correlations)

# Find the synthetic file with the highest average correlation
best_synthetic_file = max(average_correlations, key=average_correlations.get)

print("Synthetic file with the highest average correlation:")
print(f"{best_synthetic_file} with average correlation: {average_correlations[best_synthetic_file]:.4f}")

# Save results to a file
with open('PULSOVITAL/results/average_correlation_results.txt', 'w') as f:
    f.write("Synthetic file with the highest average correlation:\n")
    f.write(f"{best_synthetic_file} with average correlation: {average_correlations[best_synthetic_file]:.4f}\n\n")
    f.write("All synthetic files and their average correlations:\n")
    for file, avg_corr in average_correlations.items():
        f.write(f"{file}: {avg_corr:.4f}\n")

print("Results saved to 'PULSOVITAL/results/average_correlation_results.txt'")