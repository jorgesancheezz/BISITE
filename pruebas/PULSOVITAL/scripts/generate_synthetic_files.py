import numpy as np
import os

# Function to generate synthetic data
def generate_synthetic_data(scale, noise, length=10000, num_samples=128):
    np.random.seed(42)  # For reproducibility
    data = scale * np.random.randn(num_samples, length, 1) + noise * np.random.randn(num_samples, length, 1)
    return data

# Directory to save synthetic files
output_dir = 'PULSOVITAL/results'
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic files for scales from 0.0 to 0.10 and noise levels from 0.01 to 0.10
for scale in np.arange(0.0, 0.11, 0.01):
    for noise in np.arange(0.01, 0.11, 0.01):
        scale = round(scale, 2)  # Avoid floating-point precision issues
        noise = round(noise, 2)
        synthetic_data = generate_synthetic_data(scale, noise)
        output_file = os.path.join(output_dir, f'synthetic_scale_{scale:.2f}_noise_{noise:.2f}.npz')
        np.savez(output_file, data=synthetic_data)
        print(f"Generated: {output_file}")

print("All synthetic files generated.")