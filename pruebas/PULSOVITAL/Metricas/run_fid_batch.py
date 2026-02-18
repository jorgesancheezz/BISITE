import numpy as np
import os
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

print('run_fid_batch start', flush=True)

pairs = [
    {
        'name': 'AF',
        'real': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/AF_signals_1024.npy',
        'synth': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/1024seq_AF.npy',
        'out_csv': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/fid_results_AF.csv',
        'out_png': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/fid_plot_AF.png'
    },
    {
        'name': 'NSR',
        'real': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/NSR_signals_1024.npy',
        'synth': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/1024seq_NSR.npy',
        'out_csv': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/fid_results_NSR.csv',
        'out_png': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/fid_plot_NSR.png'
    }
]

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


def arr_to_list(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return [arr[i, :, 0].astype(float) for i in range(arr.shape[0])]
    if arr.ndim == 2:
        return [arr[i, :].astype(float) for i in range(arr.shape[0])]
    if arr.ndim == 1:
        return [arr.astype(float)]
    return [s.astype(float).ravel() for s in arr]

for p in pairs:
    try:
        print(f"Processing {p['name']}...", flush=True)
        real_arr = np.load(p['real'])
        synth_arr = np.load(p['synth'])
        print('loaded shapes', real_arr.shape, synth_arr.shape, flush=True)

        real_signals = arr_to_list(real_arr)
        synth_signals = arr_to_list(synth_arr)
        print('n signals', len(real_signals), len(synth_signals), flush=True)

        L = 1000
        real_proc = truncate_signals([normalize_signal(s) for s in real_signals], L)
        synth_proc = truncate_signals([normalize_signal(s) for s in synth_signals], L)
        print('after truncation', real_proc.shape, synth_proc.shape, flush=True)

        fid_val = calculate_fid(real_proc, synth_proc)
        print(f"{p['name']} FID: {fid_val}", flush=True)

        os.makedirs(os.path.dirname(p['out_csv']), exist_ok=True)
        import csv
        with open(p['out_csv'], 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['metric','value'])
            w.writerow(['FID', fid_val])
        print('saved csv', p['out_csv'], flush=True)

        plt.figure(figsize=(4,3))
        plt.bar(['FID'], [fid_val])
        plt.title(f"FID {p['name']}")
        plt.tight_layout()
        plt.savefig(p['out_png'])
        plt.close()
        print('saved plot', p['out_png'], flush=True)

    except Exception as e:
        import traceback
        print('error processing', p['name'], e, flush=True)
        traceback.print_exc()
