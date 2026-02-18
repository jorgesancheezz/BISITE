import numpy as np
import os
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

print('run_fid_test start', flush=True)

real_path = r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/AF_signals_1024.npy'
synth_path = r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/1024seq_AF.npy'
output_csv = r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/fid_results_test.csv'
output_plot = r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/fid_plot_test.png'

# helper funcs

def normalize_signal(sig):
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    return np.clip(sig, -5, 5)

def truncate_signals(signals, length=1000):
    return np.array([s[:length] for s in signals])

def calculate_fid(real, synth):
    mu_real = np.mean(real, axis=0)
    mu_synth = np.mean(synth, axis=0)
    cov_real = np.cov(real, rowvar=False)
    cov_synth = np.cov(synth, rowvar=False)
    cov_sqrt = sqrtm(cov_real @ cov_synth)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    fid = np.sum((mu_real - mu_synth)**2) + np.trace(cov_real + cov_synth - 2 * cov_sqrt)
    return float(fid)

try:
    real_arr = np.load(real_path)
    synth_arr = np.load(synth_path)
    print('loaded shapes', real_arr.shape, synth_arr.shape, flush=True)

    def arr_to_list(arr):
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return [arr[i, :, 0].astype(float) for i in range(arr.shape[0])]
        if arr.ndim == 2:
            return [arr[i, :].astype(float) for i in range(arr.shape[0])]
        if arr.ndim == 1:
            return [arr.astype(float)]
        return [s.astype(float).ravel() for s in arr]

    real_signals = arr_to_list(real_arr)
    synth_signals = arr_to_list(synth_arr)
    print('n signals', len(real_signals), len(synth_signals), flush=True)

    L = 1000
    real_proc = truncate_signals([normalize_signal(s) for s in real_signals], L)
    synth_proc = truncate_signals([normalize_signal(s) for s in synth_signals], L)
    print('after truncation', real_proc.shape, synth_proc.shape, flush=True)

    fid_val = calculate_fid(real_proc, synth_proc)
    print('fid', fid_val, flush=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    import csv
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric','value'])
        w.writerow(['FID', fid_val])
    print('saved csv', output_csv, flush=True)

    plt.figure(figsize=(4,3))
    plt.bar(['FID'], [fid_val])
    plt.title('FID')
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()
    print('saved plot', output_plot, flush=True)

except Exception as e:
    import traceback
    print('error', e, flush=True)
    traceback.print_exc()
