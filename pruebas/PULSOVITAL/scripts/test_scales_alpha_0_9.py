from PULSOVITAL.data.synthetic_loader import build_synthetic_loader, collect_samples
from PULSOVITAL.plotting.plot_psd_full import plot_average_spectrum, load_npy
from scipy.signal import welch
import numpy as np

# Define alphas and scales to test
alphas = [0.85, 0.9, 0.95]  # Adding more alpha values
scales = [round(x, 4) for x in np.arange(0.1150, 0.1351, 0.001)]  # Fine-grained scale adjustments
ref_file = 'PULSOVITAL/results/004.npy'

# Load reference data
ref_data = load_npy(ref_file)

# Define noise levels to test
noise_levels = [0.03, 0.05, 0.07]  # Adjusting noise slightly

# Test each combination of alpha and scale
for alpha in alphas:
    for scale in scales:
        for noise in noise_levels:
            # Build DataLoader with specified parameters
            loader = build_synthetic_loader(
                generator='wavelet',
                length=10000,
                alpha=alpha,  # Adjusted alpha value
                noise=noise,
                batch_size=4,
                num_samples=128,
                scale=scale
            )

            # Collect synthetic samples
            synth_data = collect_samples(loader, max_samples=128)

            # Generate PSD comparison plot
            labeled_datas = [("004.npy", ref_data), (f"Synthetic Alpha {alpha}, Scale {scale}, Noise {noise}", synth_data)]
            output_path = f'PULSOVITAL/results/psd_comparison_alpha_{alpha}_scale_{scale}_noise_{noise}.png'
            plot_average_spectrum(output_path, labeled_datas, fs=250.0, nperseg=256, noverlap=128, smooth=True, title=f'PSD Comparison (Alpha {alpha}, Scale {scale}, Noise {noise})')

print("PSD plots generated for various alphas, scales, and noise levels.")