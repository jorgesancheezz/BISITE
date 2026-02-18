from PULSOVITAL.plotting.plot_psd_full import plot_average_spectrum, load_npy
import numpy as np

# Define scales and reference file
scales = [0.1, 0.2, 0.3, 0.4, 0.5]
ref_file = 'PULSOVITAL/results/004.npy'

# Load reference data
ref_data = load_npy(ref_file)

# Generate PSD plots for each scale
for scale in scales:
    synth_file = f'PULSOVITAL/results/scale_{scale}_result.npz'
    synth_data = np.load(synth_file)['data']  # Extract the actual data array
    labeled_datas = [("004.npy", ref_data), (f"Synthetic Scale {scale}", synth_data)]
    output_path = f'PULSOVITAL/results/psd_comparison_scale_{scale}.png'
    plot_average_spectrum(output_path, labeled_datas, fs=250.0, nperseg=256, noverlap=128, smooth=True, title=f'PSD Comparison (Scale {scale})')

print("PSD plots generated for all scales.")