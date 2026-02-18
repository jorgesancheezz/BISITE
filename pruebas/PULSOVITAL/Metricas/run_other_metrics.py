import numpy as np
import os
from scipy.signal import welch
from scipy.spatial.distance import cdist
from scipy.stats import energy_distance
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import csv
import traceback

print('run_other_metrics start', flush=True)

pairs = [
    {
        'name': 'AF',
        'real': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/AF_signals_1024.npy',
        'synth': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/1024seq_AF.npy',
        'out_csv': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/metrics_AF.csv',
        'out_dir': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/'
    },
    {
        'name': 'NSR',
        'real': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/NSR_signals_1024.npy',
        'synth': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/1024seq_NSR.npy',
        'out_csv': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/metrics_NSR.csv',
        'out_dir': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/'
    }
]

# Helpers

def normalize_signal(sig):
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    return np.clip(sig, -5, 5)


def truncate_signals(signals, length=1000):
    return np.array([s[:length] for s in signals])


def arr_to_list(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return [arr[i, :, 0].astype(float) for i in range(arr.shape[0])]
    if arr.ndim == 2:
        return [arr[i, :].astype(float) for i in range(arr.shape[0])]
    if arr.ndim == 1:
        return [arr.astype(float)]
    return [s.astype(float).ravel() for s in arr]

# ACF

def acf(x, max_lag=200):
    result = np.correlate(x, x, mode='full')
    result = result[result.size//2:]
    if result[0] == 0:
        return np.zeros(min(max_lag, len(result)))
    return result[:max_lag] / result[0]


def evaluate_acf(real, synth, samples=50, max_lag=200):
    diffs = []
    for _ in range(samples):
        r = acf(real[np.random.randint(len(real))], max_lag=max_lag)
        s = acf(synth[np.random.randint(len(synth))], max_lag=max_lag)
        L = min(len(r), len(s))
        diffs.append(np.mean(np.abs(r[:L] - s[:L])))
    return np.mean(diffs), np.std(diffs)

# MMD

def gaussian_kernel(x, y, sigma=1.0):
    # x,y are 2D arrays
    return np.exp(-cdist(x, y, 'sqeuclidean') / (2 * sigma**2))


def compute_mmd(real, synth, sigma=1.0):
    # Ensure numeric 2D arrays
    X = np.asarray(real, dtype=float)
    Y = np.asarray(synth, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        # try to reshape
        X = np.vstack([np.ravel(x) for x in real]).astype(float)
        Y = np.vstack([np.ravel(y) for y in synth]).astype(float)

    # Truncate to common length
    L = min(X.shape[1], Y.shape[1])
    X = X[:, :L]
    Y = Y[:, :L]

    Kxx = gaussian_kernel(X, X, sigma).mean()
    Kyy = gaussian_kernel(Y, Y, sigma).mean()
    Kxy = gaussian_kernel(X, Y, sigma).mean()
    return float(Kxx + Kyy - 2 * Kxy)

# Energy distance

def evaluate_energy(real, synth, samples=200):
    n_real = len(real)
    n_synth = len(synth)
    # sample indices
    idx_r = np.random.randint(0, n_real, size=samples)
    idx_s = np.random.randint(0, n_synth, size=samples)
    R = np.array([real[i][0:500] for i in idx_r])
    S = np.array([synth[i][0:500] for i in idx_s])
    return float(energy_distance(R.flatten(), S.flatten()))

# Run for each pair
for p in pairs:
    try:
        print(f"Processing {p['name']}...", flush=True)
        real_arr = np.load(p['real'])
        synth_arr = np.load(p['synth'])
        print('loaded shapes', real_arr.shape, synth_arr.shape, flush=True)

        real_signals = arr_to_list(real_arr)
        synth_signals = arr_to_list(synth_arr)
        print('n signals', len(real_signals), len(synth_signals), flush=True)

        # preprocess and truncate
        L = 1000
        real_proc = truncate_signals([normalize_signal(s) for s in real_signals], L)
        synth_proc = truncate_signals([normalize_signal(s) for s in synth_signals], L)

        print('after truncation', real_proc.shape, synth_proc.shape, flush=True)

        # compute metrics
        acf_mean, acf_std = evaluate_acf(real_proc, synth_proc, samples=50, max_lag=200)
        print(f"ACF mean/std: {acf_mean:.6f} +/- {acf_std:.6f}", flush=True)

        mmd_val = compute_mmd(real_proc, synth_proc, sigma=1.0)
        print(f"MMD: {mmd_val}", flush=True)

        energy_val = evaluate_energy(real_proc, synth_proc, samples=200)
        print(f"Energy distance: {energy_val}", flush=True)

        # save csv
        os.makedirs(os.path.dirname(p['out_csv']), exist_ok=True)
        with open(p['out_csv'], 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['metric','value'])
            w.writerow(['ACF_mean', acf_mean])
            w.writerow(['ACF_std', acf_std])
            w.writerow(['MMD', mmd_val])
            w.writerow(['Energy', energy_val])
        print('saved csv', p['out_csv'], flush=True)

        # plots: simple bars
        try:
            # ACF bar (mean with error)
            fig, ax = plt.subplots(figsize=(4,3))
            ax.bar(['ACF'], [acf_mean], yerr=[acf_std], capsize=5)
            ax.set_title(f"ACF mean {p['name']}")
            plt.tight_layout()
            out_acf = os.path.join(p['out_dir'], f"acf_{p['name']}.png")
            plt.savefig(out_acf)
            plt.close()
            print('saved', out_acf, flush=True)

            # MMD and Energy bars
            fig, ax = plt.subplots(figsize=(5,3))
            ax.bar(['MMD','Energy'], [mmd_val, energy_val], color=['tab:orange','tab:green'])
            ax.set_title(f"MMD & Energy {p['name']}")
            plt.tight_layout()
            out_me = os.path.join(p['out_dir'], f"mmd_energy_{p['name']}.png")
            plt.savefig(out_me)
            plt.close()
            print('saved', out_me, flush=True)
        except Exception as e_plot:
            print('plot error', e_plot, flush=True)
            traceback.print_exc()

    except Exception as e:
        print('error processing', p['name'], e, flush=True)
        traceback.print_exc()
