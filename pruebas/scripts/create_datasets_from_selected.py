import os
import csv
import numpy as np
import random

# Config
selected_csv = os.path.join('PULSOVITAL', 'npy_output', 'selected_records_600.csv')
out_dir = os.path.join('PULSOVITAL', 'npy_output')
seed = 42
n_per_class_target = 1024
segment_len = 3000
channel_target = 1
mapping_csv = os.path.join(out_dir, 'selected_mapping_1024.csv')

rng = np.random.RandomState(seed)
random.seed(seed)

# Read selected
records = []
with open(selected_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        records.append(r)

by_tag = {'NSR': [], 'AF': []}
for r in records:
    tag = r.get('tag','').upper()
    if tag in by_tag:
        by_tag[tag].append(r)

print('Found selected counts -> NSR:', len(by_tag['NSR']), 'AF:', len(by_tag['AF']))

os.makedirs(out_dir, exist_ok=True)

def load_signal_from_outfolder(out_folder):
    sig_path = os.path.join(out_folder, 'signal.npy')
    if not os.path.exists(sig_path):
        raise FileNotFoundError(sig_path)
    s = np.load(sig_path)
    # s expected shape (n_samples, n_channels) or (n_samples,)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    # take first channel
    ch = s[:,0]
    return ch.astype(np.float32)


def build_array_for_tag(tag_list, target_n, rng):
    n_available = len(tag_list)
    if n_available == 0:
        raise ValueError('No records for this tag')
    replace = n_available < target_n
    indices = rng.choice(n_available, size=target_n, replace=replace)
    out_arr = np.zeros((target_n, segment_len, channel_target), dtype=np.float32)
    map_rows = []
    for i, idx in enumerate(indices):
        rec = tag_list[idx]
        out_folder = rec.get('out_folder')
        rec_name = rec.get('record')
        try:
            sig = load_signal_from_outfolder(out_folder)
            L = sig.shape[0]
            if L >= segment_len:
                start = int(rng.randint(0, L - segment_len + 1))
                seg = sig[start:start+segment_len]
            else:
                # pad zeros at end
                seg = np.zeros(segment_len, dtype=np.float32)
                seg[:L] = sig
                start = 0
            seg = seg.reshape(segment_len, 1)
            out_arr[i] = seg
            map_rows.append((i, rec_name, out_folder, L, start))
        except Exception as e:
            # If load fails, try to pick another random candidate (up to attempts)
            print(f'[WARN] failed load {out_folder}: {e}. retrying with another sample')
            # attempt replacements up to 10 tries
            success = False
            for attempt in range(10):
                j = int(rng.randint(0, n_available))
                rec2 = tag_list[j]
                try:
                    sig = load_signal_from_outfolder(rec2.get('out_folder'))
                    L = sig.shape[0]
                    if L >= segment_len:
                        start = int(rng.randint(0, L - segment_len + 1))
                        seg = sig[start:start+segment_len]
                    else:
                        seg = np.zeros(segment_len, dtype=np.float32)
                        seg[:L] = sig
                        start = 0
                    seg = seg.reshape(segment_len, 1)
                    out_arr[i] = seg
                    map_rows.append((i, rec2.get('record'), rec2.get('out_folder'), L, start))
                    success = True
                    break
                except Exception:
                    continue
            if not success:
                # leave zeros and record failure
                map_rows.append((i, rec_name, out_folder, -1, -1))
    return out_arr, map_rows

# Build NSR
print('Building NSR array...')
nsr_arr, nsr_map = build_array_for_tag(by_tag['NSR'], n_per_class_target, rng)
print('Building AF array...')
af_arr, af_map = build_array_for_tag(by_tag['AF'], n_per_class_target, rng)

# Save
af_path = os.path.join(out_dir, f'AF_selected_{n_per_class_target}x{segment_len}x{channel_target}.npy')
nsr_path = os.path.join(out_dir, f'NSR_selected_{n_per_class_target}x{segment_len}x{channel_target}.npy')
np.save(af_path, af_arr)
np.save(nsr_path, nsr_arr)

# Save mapping CSV
with open(mapping_csv, 'w', newline='', encoding='utf-8') as mf:
    writer = csv.writer(mf)
    writer.writerow(['index','tag','record','out_folder','original_length','start_index'])
    for i, rec_name, out_folder, L, start in nsr_map:
        writer.writerow([i,'NSR',rec_name,out_folder,L,start])
    for i, rec_name, out_folder, L, start in af_map:
        writer.writerow([i,'AF',rec_name,out_folder,L,start])

print('Saved arrays:')
print(' AF ->', af_path)
print(' NSR ->', nsr_path)
print(' Mapping CSV ->', mapping_csv)
print('Done.')
