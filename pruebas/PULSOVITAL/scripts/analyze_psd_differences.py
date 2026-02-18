import numpy as np

# Define the scales to analyze
scales = [0.1, 0.2, 0.3, 0.4, 0.5]

# Initialize results dictionary
results = {}

# Load and analyze PSD differences for each scale
for scale in scales:
    data = np.load(f'PULSOVITAL/results/scale_{scale}_result.npz')
    results[scale] = data['diff']

# Save consolidated results
with open('PULSOVITAL/results/psd_differences.txt', 'w') as f:
    for scale, diff in results.items():
        f.write(f'Scale: {scale}, PSD Difference: {diff}\n')

print("Analysis complete. Results saved to PULSOVITAL/results/psd_differences.txt")