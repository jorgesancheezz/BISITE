import sys
import os
import numpy as np
from PULSOVITAL.data.synthetic_loader import build_synthetic_loader, collect_samples

# Add PULSOVITAL to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Asegurar que el directorio raíz del proyecto esté en sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Parameters
length = 10000
alpha = 0.9
num_samples = 128
batch_size = 4

# Load statistics of `sintetico.npy`
sintetico_stats = {
    'mean': 7.2438364e-05,
    'std': 0.92190087,
    'min': -5.51778,
    'max': 5.627169
}

# Adjust scales and noise dynamically based on `sintetico.npy` statistics
scales = np.arange(0.1260, 0.1281, 0.0001)
noises = [0.05]  # Keep noise levels consistent

# Generate .npz files with updated scales
for scale in scales:
    for noise in noises:
        loader = build_synthetic_loader(generator='wavelet', length=length, alpha=alpha, noise=noise, num_samples=num_samples, scale=scale)
        data = collect_samples(loader, max_samples=num_samples)
        output_file = f'PULSOVITAL/results/synthetic_scale_{scale:.4f}_noise_{noise}.npz'
        np.savez(output_file, data=data, scale=scale, noise=noise)
        print(f"Generated {output_file}")