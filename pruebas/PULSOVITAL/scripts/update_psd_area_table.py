import numpy as np
from scipy.signal import welch
import csv
import os

# Load reference PSD data
ref_data = np.load('PULSOVITAL/results/ref_psd_data.npz')
ref_psd = ref_data['ref_psd']

# Define specific references to include
specific_refs = ['003.npy', '004.npy', '005.npy', '006.npy', '007.npy']

# Get list of new .npz files (filter only relevant ones)
results_dir = 'PULSOVITAL/results'
npz_files = [f for f in os.listdir(results_dir) if f.startswith('synthetic_scale_') and f.endswith('.npz') and any(ref in f for ref in specific_refs)]

# Load all synthetic .npz files as additional reference PSDs
synthetic_refs = {}
for ref_file in npz_files:
    ref_path = os.path.join(results_dir, ref_file)
    ref_data = np.load(ref_path)
    synthetic_refs[ref_file] = ref_data['data'][:, :, 0].mean(axis=0)

# Load existing CSV data
csv_file = 'PULSOVITAL/results/psd_area_differences.csv'
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)

header = rows[0]
data_rows = rows[1:]

# Update table with new .npz files
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Synthetic'] + specific_refs)  # Ensure correct header

    # Write existing rows (filtered by specific references)
    for row in data_rows:
        filtered_row = [row[0]] + [row[header.index(ref)] for ref in specific_refs if ref in header]
        writer.writerow(filtered_row)

    # Process new .npz files
    for npz_file in npz_files:
        file_path = os.path.join(results_dir, npz_file)
        data = np.load(file_path)
        synth_data = data['data']

        # Calculate PSD
        synth_freqs, synth_psd = welch(synth_data[:, :, 0].mean(axis=0), fs=250.0, nperseg=256, noverlap=128)

        # Calculate differences for each specific reference PSD
        diffs = []
        for ref_name in specific_refs:
            if ref_name in synthetic_refs:
                ref_psd_data = synthetic_refs[ref_name]
                ref_freqs, ref_psd = welch(ref_psd_data, fs=250.0, nperseg=256, noverlap=128)
                diffs.append(np.mean(np.abs(ref_psd - synth_psd)))

        # Write new row
        writer.writerow([npz_file] + diffs)

print("Table updated successfully.")