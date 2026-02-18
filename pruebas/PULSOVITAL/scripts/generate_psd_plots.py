from PULSOVITAL.data.synthetic_loader import build_synthetic_loader, collect_samples
from PULSOVITAL.plotting.plot_psd_full import plot_average_spectrum, load_npy
from scipy.signal import welch
from scipy.spatial.distance import jensenshannon
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Load reference data
ref_file = 'PULSOVITAL/results/004.npy'
ref_data = load_npy(ref_file)

# Load statistics of `sintetico.npy`
sintetico_stats = {
    'mean': 7.2438364e-05,
    'std': 0.92190087,
    'min': -5.51778,
    'max': 5.627169
}

# Adjust scales and noise dynamically based on `sintetico.npy` statistics
scales = [sintetico_stats['std'] * factor for factor in [0.8, 0.9, 1.0, 1.1, 1.2]]
noise = 0.05  # Keep noise constant for now

# Generate PSD plots for each scale
for scale in scales:
    # Generate synthetic data with adjusted parameters
    loader = build_synthetic_loader(generator='wavelet', length=10000, alpha=0.9, noise=noise, num_samples=128, scale=scale)
    synth_data = collect_samples(loader, max_samples=128)

    # Check data length and handle short signals
    if len(ref_data) < 2 or len(synth_data) < 2:
        print(f"Skipping scale {scale} due to insufficient data length.")
        continue

    # Debugging: Print the shape of the reference and synthetic data
    print(f"Shape of ref_data: {ref_data.shape}")
    print(f"Shape of synth_data: {synth_data.shape}")

    # Adjust mean calculation to preserve signal length
    ref_signal = ref_data.mean(axis=(0, 2))
    synth_signal = synth_data.mean(axis=(0, 2))

    # Debugging: Save the reference and synthetic signals for inspection
    np.save(f'PULSOVITAL/results/debug_ref_signal.npy', ref_signal)
    np.save(f'PULSOVITAL/results/debug_synth_signal_scale_{scale:.2f}.npy', synth_signal)

    # Calculate PSD for reference and synthetic data
    ref_freqs, ref_psd = welch(ref_signal, fs=250.0, nperseg=256, noverlap=128)
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)

    # Plot PSD comparison
    plt.figure(figsize=(12, 6))
    plt.plot(ref_freqs, 10 * np.log10(ref_psd), label='004.npy', color='blue')
    plt.plot(synth_freqs, 10 * np.log10(synth_psd), label=f'Synthetic Scale {scale:.2f}', color='orange')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Potencia (dB/Hz)')
    plt.title(f'PSD medio (completo): 004.npy vs Synthetic Scale {scale:.2f}')
    plt.legend()
    plt.grid(True)
    output_path = f'PULSOVITAL/results/psd_comparison_scale_{scale:.2f}.png'
    plt.savefig(output_path)
    plt.close()

# Load all synthetic .npz files
synthetic_files = glob.glob('PULSOVITAL/results/synthetic_scale_*.npz')

# Filter synthetic files based on PSD threshold
threshold_db = -80
synthetic_files = glob.glob('PULSOVITAL/results/synthetic_scale_*.npz')

for synthetic_file in synthetic_files:
    data = np.load(synthetic_file)['data']
    synth_signal = data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    max_psd_db = 10 * np.log10(synth_psd).max()

    if max_psd_db < threshold_db:
        os.remove(synthetic_file)
        print(f"Deleted {synthetic_file} due to PSD below threshold ({max_psd_db:.2f} dB/Hz)")

# Generate PSD comparison plot
plt.figure(figsize=(12, 6))

# Calculate PSD for reference data
ref_signal = ref_data.mean(axis=(0, 2))
ref_freqs, ref_psd = welch(ref_signal, fs=250.0, nperseg=256, noverlap=128)
plt.plot(ref_freqs, 10 * np.log10(ref_psd), label='004.npy (Reference)', color='blue')

# Loop through synthetic files and calculate PSD
for synthetic_file in synthetic_files:
    data = np.load(synthetic_file)['data']
    synth_signal = data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    label = synthetic_file.split('/')[-1].replace('synthetic_', '').replace('.npz', '')
    plt.plot(synth_freqs, 10 * np.log10(synth_psd), label=label)

# Finalize plot
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia (dB/Hz)')
plt.title('Comparación de PSD: Referencia vs Sintéticos')
plt.legend()
plt.grid(True)
plt.savefig('PULSOVITAL/results/psd_comparison_all.png')
plt.close()

# Generate PSD comparison plot excluding deleted files
remaining_files = glob.glob('PULSOVITAL/results/synthetic_scale_*.npz')

plt.figure(figsize=(12, 6))

# Calculate PSD for reference data
ref_signal = ref_data.mean(axis=(0, 2))
ref_freqs, ref_psd = welch(ref_signal, fs=250.0, nperseg=256, noverlap=128)
plt.plot(ref_freqs, 10 * np.log10(ref_psd), label='004.npy (Reference)', color='blue')

# Loop through remaining synthetic files and calculate PSD
for synthetic_file in remaining_files:
    data = np.load(synthetic_file)['data']
    synth_signal = data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    label = synthetic_file.split('/')[-1].replace('synthetic_', '').replace('.npz', '')
    plt.plot(synth_freqs, 10 * np.log10(synth_psd), label=label)

# Finalize plot
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia (dB/Hz)')
plt.title('Comparación de PSD: Referencia vs Sintéticos (Excluyendo eliminados)')
plt.legend()
plt.grid(True)
plt.savefig('PULSOVITAL/results/psd_comparison_filtered.png')
plt.close()

# Calculate similarity between PSDs
similarities = []
remaining_files = glob.glob('PULSOVITAL/results/synthetic_scale_*.npz')

# Calculate PSD for reference data
ref_signal = ref_data.mean(axis=(0, 2))
ref_freqs, ref_psd = welch(ref_signal, fs=250.0, nperseg=256, noverlap=128)
ref_psd_db = 10 * np.log10(ref_psd)

# Loop through remaining synthetic files and calculate similarity
for synthetic_file in remaining_files:
    data = np.load(synthetic_file)['data']
    synth_signal = data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    synth_psd_db = 10 * np.log10(synth_psd)

    # Calculate Euclidean distance as similarity metric
    distance = np.linalg.norm(ref_psd_db - synth_psd_db)
    similarities.append((synthetic_file, distance))

# Find the most similar synthetic file
most_similar = min(similarities, key=lambda x: x[1])
print(f"Most similar synthetic file: {most_similar[0]} with distance: {most_similar[1]:.2f}")

# Generate PSD comparison plot for reference, most similar synthetic, and `sintetico.npy`
sintetico_data = np.load('PULSOVITAL/results/sintetico.npy')
sintetico_signal = sintetico_data.mean(axis=(0, 2))
sintetico_freqs, sintetico_psd = welch(sintetico_signal, fs=250.0, nperseg=256, noverlap=128)
sintetico_psd_db = 10 * np.log10(sintetico_psd)

most_similar_file = 'PULSOVITAL/results/synthetic_scale_0.1190_noise_0.1.npz'
most_similar_data = np.load(most_similar_file)['data']
most_similar_signal = most_similar_data.mean(axis=(0, 2))
most_similar_freqs, most_similar_psd = welch(most_similar_signal, fs=250.0, nperseg=256, noverlap=128)
most_similar_psd_db = 10 * np.log10(most_similar_psd)

plt.figure(figsize=(12, 6))

# Plot reference PSD
plt.plot(ref_freqs, ref_psd_db, label='004.npy (Reference)', color='blue')

# Plot most similar synthetic PSD
plt.plot(most_similar_freqs, most_similar_psd_db, label='Most Similar Synthetic', color='orange')

# Plot `sintetico.npy` PSD
plt.plot(sintetico_freqs, sintetico_psd_db, label='sintetico.npy', color='green')

# Finalize plot
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia (dB/Hz)')
plt.title('Comparación de PSD: Referencia, Sintético más similar y `sintetico.npy`')
plt.legend()
plt.grid(True)
plt.savefig('PULSOVITAL/results/psd_comparison_selected.png')
plt.close()

# Load reference data for `003.npy`
ref_file_003 = 'PULSOVITAL/results/003.npy'
ref_data_003 = load_npy(ref_file_003)
ref_signal_003 = ref_data_003.mean(axis=(0, 2))
ref_freqs_003, ref_psd_003 = welch(ref_signal_003, fs=250.0, nperseg=256, noverlap=128)
ref_psd_db_003 = 10 * np.log10(ref_psd_003)

plt.figure(figsize=(12, 6))

# Plot PSD for `004.npy`
plt.plot(ref_freqs, ref_psd_db, label='004.npy (Reference)', color='blue')

# Plot PSD for `003.npy`
plt.plot(ref_freqs_003, ref_psd_db_003, label='003.npy (Reference)', color='red')

# Plot most similar synthetic PSD
plt.plot(most_similar_freqs, most_similar_psd_db, label='Most Similar Synthetic', color='orange')

# Plot `sintetico.npy` PSD
plt.plot(sintetico_freqs, sintetico_psd_db, label='sintetico.npy', color='green')

# Finalize plot
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia (dB/Hz)')
plt.title('Comparación de PSD: Referencias (003 y 004), Sintético más similar y `sintetico.npy`')
plt.legend()
plt.grid(True)
plt.savefig('PULSOVITAL/results/psd_comparison_with_003.png')
plt.close()

# Calculate similarity between PSDs for `003.npy`
similarities_003 = []

# Calculate PSD for `003.npy`
ref_signal_003 = ref_data_003.mean(axis=(0, 2))
ref_freqs_003, ref_psd_003 = welch(ref_signal_003, fs=250.0, nperseg=256, noverlap=128)
ref_psd_db_003 = 10 * np.log10(ref_psd_003)

# Loop through remaining synthetic files and calculate similarity
for synthetic_file in remaining_files:
    data = np.load(synthetic_file)['data']
    synth_signal = data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    synth_psd_db = 10 * np.log10(synth_psd)

    # Calculate Euclidean distance as similarity metric
    distance = np.linalg.norm(ref_psd_db_003 - synth_psd_db)
    similarities_003.append((synthetic_file, distance))

# Find the most similar synthetic file to `003.npy`
most_similar_003 = min(similarities_003, key=lambda x: x[1])
print(f"Most similar synthetic file to `003.npy`: {most_similar_003[0]} with distance: {most_similar_003[1]:.2f}")

# Load reference data for `006.npy` and `007.npy`
ref_file_006 = 'PULSOVITAL/results/006.npy'
ref_data_006 = load_npy(ref_file_006)
ref_signal_006 = ref_data_006.mean(axis=(0, 2))
ref_freqs_006, ref_psd_006 = welch(ref_signal_006, fs=250.0, nperseg=256, noverlap=128)
ref_psd_db_006 = 10 * np.log10(ref_psd_006)

ref_file_007 = 'PULSOVITAL/results/007.npy'
ref_data_007 = load_npy(ref_file_007)
ref_signal_007 = ref_data_007.mean(axis=(0, 2))
ref_freqs_007, ref_psd_007 = welch(ref_signal_007, fs=250.0, nperseg=256, noverlap=128)
ref_psd_db_007 = 10 * np.log10(ref_psd_007)

# Calculate similarity for `006.npy`
similarities_006 = []
for synthetic_file in remaining_files:
    data = np.load(synthetic_file)['data']
    synth_signal = data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    synth_psd_db = 10 * np.log10(synth_psd)
    distance = np.linalg.norm(ref_psd_db_006 - synth_psd_db)
    similarities_006.append((synthetic_file, distance))
most_similar_006 = min(similarities_006, key=lambda x: x[1])
print(f"Most similar synthetic file to `006.npy`: {most_similar_006[0]} with distance: {most_similar_006[1]:.2f}")

# Calculate similarity for `007.npy`
similarities_007 = []
for synthetic_file in remaining_files:
    data = np.load(synthetic_file)['data']
    synth_signal = data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    synth_psd_db = 10 * np.log10(synth_psd)
    distance = np.linalg.norm(ref_psd_db_007 - synth_psd_db)
    similarities_007.append((synthetic_file, distance))
most_similar_007 = min(similarities_007, key=lambda x: x[1])
print(f"Most similar synthetic file to `007.npy`: {most_similar_007[0]} with distance: {most_similar_007[1]:.2f}")

# Generate PSD comparison plot for selected synthetic files
selected_files = [
    'PULSOVITAL/results/synthetic_scale_0.1190_noise_0.1.npz',
    'PULSOVITAL/results/synthetic_scale_0.1010_noise_0.1.npz',
    'PULSOVITAL/results/synthetic_scale_0.1270_noise_0.05.npz'
]

plt.figure(figsize=(12, 6))

# Plot PSD for `003.npy`
plt.plot(ref_freqs_003, ref_psd_db_003, label='003.npy (Reference)', color='red')

# Plot PSD for `004.npy`
plt.plot(ref_freqs, ref_psd_db, label='004.npy (Reference)', color='blue')

# Plot PSD for `006.npy`
plt.plot(ref_freqs_006, ref_psd_db_006, label='006.npy (Reference)', color='purple')

# Plot PSD for `007.npy`
plt.plot(ref_freqs_007, ref_psd_db_007, label='007.npy (Reference)', color='brown')

# Loop through selected synthetic files and plot their PSD
for synthetic_file in selected_files:
    data = np.load(synthetic_file)['data']
    synth_signal = data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    synth_psd_db = 10 * np.log10(synth_psd)
    label = synthetic_file.split('/')[-1].replace('synthetic_', '').replace('.npz', '')
    plt.plot(synth_freqs, synth_psd_db, label=label)

# Finalize plot
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia (dB/Hz)')
plt.title('Comparación de PSD: Referencias y Sintéticos seleccionados')
plt.legend()
plt.grid(True)
plt.savefig('PULSOVITAL/results/psd_comparison_selected_7.png')
plt.close()

# Load reference data for `005.npy`
ref_file_005 = 'PULSOVITAL/results/005.npy'
ref_data_005 = load_npy(ref_file_005)
ref_signal_005 = ref_data_005.mean(axis=(0, 2))
ref_freqs_005, ref_psd_005 = welch(ref_signal_005, fs=250.0, nperseg=256, noverlap=128)
ref_psd_db_005 = 10 * np.log10(ref_psd_005)

# Load all reference files
reference_files = [
    'PULSOVITAL/results/003.npy',
    'PULSOVITAL/results/004.npy',
    'PULSOVITAL/results/005.npy',
    'PULSOVITAL/results/006.npy',
    'PULSOVITAL/results/007.npy'
]

# Calculate PSD for all reference files
reference_psds = []
for ref_file in reference_files:
    ref_data = np.load(ref_file)
    ref_signal = ref_data.mean(axis=(0, 2))
    ref_freqs, ref_psd = welch(ref_signal, fs=250.0, nperseg=256, noverlap=128)
    reference_psds.append(10 * np.log10(ref_psd))

# Calculate average PSD across all references
average_ref_psd = np.mean(reference_psds, axis=0)

# Find the synthetic file most similar to the average reference PSD
synthetic_files = glob.glob('PULSOVITAL/results/synthetic_scale_*.npz')
min_distance = float('inf')
best_synthetic_file = None

for synthetic_file in synthetic_files:
    synth_data = np.load(synthetic_file)['data']
    synth_signal = synth_data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    synth_psd_db = 10 * np.log10(synth_psd)

    # Calculate Euclidean distance to the average reference PSD
    distance = np.linalg.norm(synth_psd_db - average_ref_psd)
    if distance < min_distance:
        min_distance = distance
        best_synthetic_file = synthetic_file

print(f"Most similar synthetic file to all references: {best_synthetic_file} with distance: {min_distance}")

# Plot PSD comparison for the best synthetic file
best_data = np.load(best_synthetic_file)['data']
best_signal = best_data.mean(axis=(0, 2))
best_freqs, best_psd = welch(best_signal, fs=250.0, nperseg=256, noverlap=128)

plt.figure(figsize=(12, 6))
plt.plot(ref_freqs, average_ref_psd, label='Average Reference PSD', color='blue')
plt.plot(best_freqs, 10 * np.log10(best_psd), label='Best Synthetic PSD', color='orange')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB/Hz)')
plt.title('PSD Comparison: Average Reference vs Best Synthetic')
plt.legend()
plt.grid(True)
plt.savefig('PULSOVITAL/results/psd_comparison_best_average.png')
plt.close()

# Plot PSD comparison for all references and the best synthetic file
plt.figure(figsize=(12, 6))

# Plot PSDs for all reference files
for i, ref_psd in enumerate(reference_psds):
    plt.plot(ref_freqs, ref_psd, label=f'Reference {reference_files[i].split("/")[-1]}')

# Plot PSD for the best synthetic file
plt.plot(best_freqs, 10 * np.log10(best_psd), label='Best Synthetic PSD', color='orange')

# Finalize the plot
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB/Hz)')
plt.title('PSD Comparison: All References vs Best Synthetic')
plt.legend()
plt.grid(True)
plt.savefig('PULSOVITAL/results/psd_comparison_all_references.png')
plt.close()

# Calculate average similarity across all references
references = [
    ('003.npy', ref_psd_db_003),
    ('004.npy', ref_psd_db),
    ('005.npy', ref_psd_db_005),
    ('006.npy', ref_psd_db_006),
    ('007.npy', ref_psd_db_007)
]

average_similarities = []
for synthetic_file in remaining_files:
    data = np.load(synthetic_file)['data']
    synth_signal = data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    synth_psd_db = 10 * np.log10(synth_psd)

    # Calculate average distance
    total_distance = 0
    for ref_name, ref_psd_db in references:
        distance = np.linalg.norm(ref_psd_db - synth_psd_db)
        total_distance += distance
    average_distance = total_distance / len(references)
    average_similarities.append((synthetic_file, average_distance))

# Find the synthetic file with the lowest average distance
most_similar_average = min(average_similarities, key=lambda x: x[1])
print(f"Most similar synthetic file to all references: {most_similar_average[0]} with average distance: {most_similar_average[1]:.2f}")

# Calculate average similarity across all references for all `.npz` files
all_npz_files = glob.glob('PULSOVITAL/results/*.npz')

average_similarities_all = []
for synthetic_file in all_npz_files:
    try:
        data = np.load(synthetic_file)['data']
        synth_signal = data.mean(axis=(0, 2))
        synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
        synth_psd_db = 10 * np.log10(synth_psd)

        # Calculate average distance
        total_distance = 0
        for ref_name, ref_psd_db in references:
            distance = np.linalg.norm(ref_psd_db - synth_psd_db)
            total_distance += distance
        average_distance = total_distance / len(references)
        average_similarities_all.append((synthetic_file, average_distance))
    except KeyError:
        print(f"Skipping {synthetic_file} due to missing 'data' key.")

# Find the synthetic file with the lowest average distance
most_similar_average_all = min(average_similarities_all, key=lambda x: x[1])
print(f"Most similar synthetic file to all references (all `.npz` files): {most_similar_average_all[0]} with average distance: {most_similar_average_all[1]:.2f}")

# Calculate average similarity for new `.npz` files in range 0.1270 to 0.1280
new_files = glob.glob('PULSOVITAL/results/synthetic_scale_0.127*_noise_0.05.npz')

average_similarities_new = []
for synthetic_file in new_files:
    data = np.load(synthetic_file)['data']
    synth_signal = data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    synth_psd_db = 10 * np.log10(synth_psd)

    # Calculate average distance
    total_distance = 0
    for ref_name, ref_psd_db in references:
        distance = np.linalg.norm(ref_psd_db - synth_psd_db)
        total_distance += distance
    average_distance = total_distance / len(references)
    average_similarities_new.append((synthetic_file, average_distance))

# Debugging: Print entries in `average_similarities_new` to identify problematic values
print("Debugging average_similarities_new:", average_similarities_new)

# Define `original_file` explicitly as the most similar synthetic file identified earlier
original_file = 'PULSOVITAL/results/synthetic_scale_0.1270_noise_0.05.npz'  # Example file, adjust as needed

# Normalize file paths for comparison
normalized_average_similarities_new = [(os.path.normpath(file), dist) for file, dist in average_similarities_new]
normalized_original_file = os.path.normpath(original_file)

# Ensure `original_distance` is calculated from the most similar synthetic file
original_distance = next((dist for file, dist in normalized_average_similarities_new if file == normalized_original_file), None)
if original_distance is None:
    raise ValueError(f"Original file {original_file} not found in average_similarities_new.")

# Compare distances
better_files = [f for f in average_similarities_new if f[1] is not None and isinstance(f[1], float) and f[1] < original_distance]
print(f"Original file average distance: {original_distance:.2f}")
print("Files with better average distance:")
for file, distance in better_files:
    print(f"{file}: {distance:.2f}")

print("PSD plots generated for all scales.")
print("PSD comparison plot generated: PULSOVITAL/results/psd_comparison_all.png")
print("Filtered PSD comparison plot generated: PULSOVITAL/results/psd_comparison_filtered.png")
print("Selected PSD comparison plot generated: PULSOVITAL/results/psd_comparison_selected.png")
print("PSD comparison plot with `003.npy` generated: PULSOVITAL/results/psd_comparison_with_003.png")
print("PSD comparison plot for selected files generated: PULSOVITAL/results/psd_comparison_selected_7.png")

# Load the best synthetic file
best_synthetic_file = 'PULSOVITAL/results/synthetic_scale_0.1240_noise_0.1.npz'
best_data = np.load(best_synthetic_file)['data']
best_signal = best_data.mean(axis=(0, 2))

# Calculate PSD for the best synthetic file
best_freqs, best_psd = welch(best_signal, fs=250.0, nperseg=256, noverlap=128)

# Plot PSD comparison for the best synthetic file
plt.figure(figsize=(12, 6))
plt.plot(ref_freqs, 10 * np.log10(ref_psd), label='Reference (004.npy)', color='blue')
plt.plot(best_freqs, 10 * np.log10(best_psd), label='Best Synthetic (Scale 0.1240, Noise 0.1)', color='orange')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB/Hz)')
plt.title('PSD Comparison: Reference vs Best Synthetic')
plt.legend()
plt.grid(True)
plt.savefig('PULSOVITAL/results/psd_comparison_best.png')
plt.close()

# Find the top 10 synthetic files most similar to the average reference PSD
synthetic_files = glob.glob('PULSOVITAL/results/synthetic_scale_*.npz')
distances = []

for synthetic_file in synthetic_files:
    synth_data = np.load(synthetic_file)['data']
    synth_signal = synth_data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    synth_psd_db = 10 * np.log10(synth_psd)

    # Calculate Euclidean distance to the average reference PSD
    distance = np.linalg.norm(synth_psd_db - average_ref_psd)
    distances.append((synthetic_file, distance))

# Sort synthetic files by distance
sorted_distances = sorted(distances, key=lambda x: x[1])

# Print the top 10 most similar synthetic files
print("Top 10 most similar synthetic files to the average reference PSD:")
for i, (file, distance) in enumerate(sorted_distances[:10], start=1):
    print(f"{i}. {file} with distance: {distance}")

# Calculate distances using a more accurate method (integral of absolute differences)
def calculate_distance(psd1, psd2, freqs):
    # Ensure the PSDs are of the same length
    assert len(psd1) == len(psd2) == len(freqs), "PSDs and frequencies must have the same length"

    # Calculate the absolute difference between the PSDs
    abs_diff = np.abs(psd1 - psd2)

    # Integrate the absolute difference over the frequency range
    distance = np.trapz(abs_diff, freqs)
    return distance

# Calculate distances for all synthetic files using the new method
distances = []
for synthetic_file in synthetic_files:
    synth_data = np.load(synthetic_file)['data']
    synth_signal = synth_data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)
    synth_psd_db = 10 * np.log10(synth_psd)

    # Calculate distance to the average reference PSD
    distance = calculate_distance(synth_psd_db, average_ref_psd, synth_freqs)
    distances.append((synthetic_file, distance))

# Sort synthetic files by the new distance metric
sorted_distances = sorted(distances, key=lambda x: x[1])

# Print the top 10 most similar synthetic files using the new method
print("Top 10 most similar synthetic files to the average reference PSD (using integral of absolute differences):")
for i, (file, distance) in enumerate(sorted_distances[:10], start=1):
    print(f"{i}. {file} with distance: {distance}")

# Calculate distances using Jensen-Shannon divergence
def calculate_js_distance(psd1, psd2):
    # Normalize PSDs to sum to 1 (convert to probability distributions)
    psd1_normalized = psd1 / np.sum(psd1)
    psd2_normalized = psd2 / np.sum(psd2)

    # Calculate Jensen-Shannon distance
    distance = jensenshannon(psd1_normalized, psd2_normalized)
    return distance

# Calculate distances for all synthetic files using Jensen-Shannon distance
distances = []
for synthetic_file in synthetic_files:
    synth_data = np.load(synthetic_file)['data']
    synth_signal = synth_data.mean(axis=(0, 2))
    synth_freqs, synth_psd = welch(synth_signal, fs=250.0, nperseg=256, noverlap=128)

    # Calculate distance to the average reference PSD
    distance = calculate_js_distance(synth_psd, average_ref_psd)
    distances.append((synthetic_file, distance))

# Sort synthetic files by the new distance metric
sorted_distances = sorted(distances, key=lambda x: x[1])

# Print the top 10 most similar synthetic files using Jensen-Shannon distance
print("Top 10 most similar synthetic files to the average reference PSD (using Jensen-Shannon distance):")
for i, (file, distance) in enumerate(sorted_distances[:10], start=1):
    print(f"{i}. {file} with distance: {distance}")

# Plot PSD of the selected synthetic file alongside reference files
selected_file = 'PULSOVITAL/results/synthetic_scale_0.1290_noise_0.1.npz'

# Load the selected synthetic file
data_selected = np.load(selected_file)['data']
signal_selected = data_selected.mean(axis=(0, 2))
freqs_selected, psd_selected = welch(signal_selected, fs=250.0, nperseg=256, noverlap=128)

# Load reference files
reference_files = [
    'PULSOVITAL/results/003.npy',
    'PULSOVITAL/results/004.npy',
    'PULSOVITAL/results/005.npy',
    'PULSOVITAL/results/006.npy',
    'PULSOVITAL/results/007.npy'
]

plt.figure(figsize=(12, 6))

# Plot PSDs for reference files
for ref_file in reference_files:
    ref_data = np.load(ref_file)
    ref_signal = ref_data.mean(axis=(0, 2))
    freqs_ref, psd_ref = welch(ref_signal, fs=250.0, nperseg=256, noverlap=128)
    plt.plot(freqs_ref, 10 * np.log10(psd_ref), label=f'Reference {ref_file.split("/")[-1]}')

# Plot PSD for the selected synthetic file
plt.plot(freqs_selected, 10 * np.log10(psd_selected), label='Selected Synthetic', color='orange')

# Finalize the plot
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB/Hz)')
plt.title('PSD Comparison: Selected Synthetic vs References')
plt.legend()
plt.grid(True)
plt.savefig('PULSOVITAL/results/psd_comparison_selected_vs_references.png')
plt.close()