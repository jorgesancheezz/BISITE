import numpy as np
from scipy.signal import welch
import glob

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

# Function to calculate average distance
def calculate_average_distance(file_path):
    data = np.load(file_path)['data']
    signal = data.mean(axis=(0, 2))
    _, psd = welch(signal, fs=250.0, nperseg=256, noverlap=128)
    avg_distance = np.mean([np.linalg.norm(psd - ref_psd) for ref_psd in ref_psds])
    return avg_distance

# Compare all synthetic files
synthetic_files = glob.glob('PULSOVITAL/results/synthetic_scale_*.npz')
results = []
for synthetic_file in synthetic_files:
    avg_distance = calculate_average_distance(synthetic_file)
    results.append((synthetic_file, avg_distance))

# Sort results by average distance
results.sort(key=lambda x: x[1])

# Print results
print("Average distances to references:")
for file, distance in results:
    print(f"{file}: {distance}")

# Best file
best_file, best_distance = results[0]
print(f"\nBest file: {best_file} with average distance: {best_distance}")