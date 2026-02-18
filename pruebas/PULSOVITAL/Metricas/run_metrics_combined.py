import numpy as np
import os
from scipy.linalg import sqrtm
from scipy.signal import welch
from scipy.spatial.distance import cdist
from scipy.stats import energy_distance, ks_2samp
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import traceback

print('run_metrics_combined start', flush=True)

pairs = [
    {
        'name': 'AF',
        'real': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/AF_signals_1024.npy',
        'synth': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/1024seq_AF.npy'
    },
    {
        'name': 'NSR',
        'real': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/NSR_signals_1024.npy',
        'synth': r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/1024seq_NSR.npy'
    }
]

output_csv = r'c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/metrics_combined.csv'

# Performance params
MAX_SE_SIGNALS = 200
SE_LENGTH = 500
PCA_COMPONENTS = 50
N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))

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


# Sample Entropy implementation
def sample_entropy(signal, m=2, r=None):
    x = np.asarray(signal, dtype=float)
    N = len(x)
    if r is None:
        r = 0.2 * np.std(x)
    def _phi(m):
        count = 0
        for i in range(N - m + 1):
            template = x[i:i+m]
            for j in range(i+1, N - m + 1):
                if np.max(np.abs(template - x[j:j+m])) <= r:
                    count += 1
        return count
    try:
        B = _phi(m)
        A = _phi(m+1)
        if B == 0:
            return np.inf
        return -np.log(A / B) if A > 0 else np.inf
    except Exception:
        return np.inf


# PSD band power helper
def band_power_ratio(sig, fs=250, bands=((0.5,4),(4,15))):
    f, Pxx = welch(sig, fs=fs, nperseg=256)
    total = np.trapz(Pxx, f)
    ratios = []
    for (a,b) in bands:
        idx = np.logical_and(f>=a, f<=b)
        power = np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0
        ratios.append(power / total if total>0 else 0.0)
    return ratios


# Worker helpers defined at module level so they are picklable by joblib
def se_worker(sig):
    return sample_entropy(sig[:SE_LENGTH])


def psd_worker(sig):
    return band_power_ratio(sig, fs=250, bands=((0.5,4),(4,15)))

# ACF

def acf(x, max_lag=200):
    result = np.correlate(x, x, mode='full')
    result = result[result.size//2:]
    if result.size == 0 or result[0] == 0:
        return np.zeros(min(max_lag, len(result)))
    return result[:max_lag] / result[0]


def evaluate_acf(real, synth, samples=50, max_lag=200):
    diffs = []
    for _ in range(samples):
        r = acf(real[np.random.randint(len(real))], max_lag=max_lag)
        s = acf(synth[np.random.randint(len(synth))], max_lag=max_lag)
        L = min(len(r), len(s))
        diffs.append(np.mean(np.abs(r[:L] - s[:L])))
    return float(np.mean(diffs)), float(np.std(diffs))

# MMD

def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-cdist(x, y, 'sqeuclidean') / (2 * sigma**2))


def compute_mmd(real, synth, sigma=1.0):
    # Ensure numeric 2D arrays
    X = np.asarray(real, dtype=float)
    Y = np.asarray(synth, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        X = np.vstack([np.ravel(x) for x in real]).astype(float)
        Y = np.vstack([np.ravel(y) for y in synth]).astype(float)
    # Truncate to common length
    L = min(X.shape[1], Y.shape[1])
    X = X[:, :L]
    Y = Y[:, :L]

    # Dimensionality reduction for speed
    if PCA_COMPONENTS is not None and PCA_COMPONENTS > 0 and X.shape[1] > PCA_COMPONENTS:
        pca = PCA(n_components=min(PCA_COMPONENTS, X.shape[1]))
        XY = np.vstack([X, Y])
        pca.fit(XY)
        X = pca.transform(X)
        Y = pca.transform(Y)

    Kxx = gaussian_kernel(X, X, sigma).mean()
    Kyy = gaussian_kernel(Y, Y, sigma).mean()
    Kxy = gaussian_kernel(X, Y, sigma).mean()
    return float(Kxx + Kyy - 2 * Kxy)

# Energy distance

def evaluate_energy(real, synth, samples=200):
    n_real = len(real)
    n_synth = len(synth)
    idx_r = np.random.randint(0, n_real, size=samples)
    idx_s = np.random.randint(0, n_synth, size=samples)
    R = np.array([real[i][0:500] for i in idx_r])
    S = np.array([synth[i][0:500] for i in idx_s])
    return float(energy_distance(R.flatten(), S.flatten()))

# FID (numpy)

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

# Main
results = []
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

        acf_mean, acf_std = evaluate_acf(real_proc, synth_proc, samples=50, max_lag=200)
        mmd_val = compute_mmd(real_proc, synth_proc, sigma=1.0)
        energy_val = evaluate_energy(real_proc, synth_proc, samples=200)
        fid_val = calculate_fid(real_proc, synth_proc)

        # New metrics
        # Sample entropy: compute on a subset and shorter segments in parallel
        idx_real = np.random.choice(len(real_proc), min(len(real_proc), MAX_SE_SIGNALS), replace=False)
        idx_synth = np.random.choice(len(synth_proc), min(len(synth_proc), MAX_SE_SIGNALS), replace=False)
        def se_worker(sig):
            return sample_entropy(sig[:SE_LENGTH])
        se_real = Parallel(n_jobs=N_JOBS)(delayed(se_worker)(real_proc[i]) for i in idx_real)
        se_synth = Parallel(n_jobs=N_JOBS)(delayed(se_worker)(synth_proc[i]) for i in idx_synth)
        se_real_mu = float(np.mean([v for v in se_real if np.isfinite(v)]) if len(se_real)>0 else np.nan)
        se_synth_mu = float(np.mean([v for v in se_synth if np.isfinite(v)]) if len(se_synth)>0 else np.nan)

        # KS test on amplitude distributions (flattened)
        real_flat = real_proc.flatten()
        synth_flat = synth_proc.flatten()
        ks_stat, ks_p = ks_2samp(real_flat, synth_flat)

        # Time-domain stats (on flattened distributions)
        mean_real = float(np.mean(real_flat))
        mean_synth = float(np.mean(synth_flat))
        std_real = float(np.std(real_flat))
        std_synth = float(np.std(synth_flat))
        skew_real = float(skew(real_flat))
        skew_synth = float(skew(synth_flat))
        kurt_real = float(kurtosis(real_flat))
        kurt_synth = float(kurtosis(synth_flat))
        median_real = float(np.median(real_flat))
        median_synth = float(np.median(synth_flat))
        iqr_real = float(np.percentile(real_flat,75) - np.percentile(real_flat,25))
        iqr_synth = float(np.percentile(synth_flat,75) - np.percentile(synth_flat,25))

        # PSD band-power ratios (average across signals)
        low_ratios_real = []
        mid_ratios_real = []
        low_ratios_synth = []
        mid_ratios_synth = []
        # PSD band-power with parallel workers
        def psd_worker(sig):
            return band_power_ratio(sig, fs=250, bands=((0.5,4),(4,15)))
        real_psd = Parallel(n_jobs=N_JOBS)(delayed(psd_worker)(s) for s in real_proc)
        synth_psd = Parallel(n_jobs=N_JOBS)(delayed(psd_worker)(s) for s in synth_proc)
        for lr, mr in real_psd:
            low_ratios_real.append(lr); mid_ratios_real.append(mr)
        for lr, mr in synth_psd:
            low_ratios_synth.append(lr); mid_ratios_synth.append(mr)
        low_real_mu = float(np.mean(low_ratios_real))
        mid_real_mu = float(np.mean(mid_ratios_real))
        low_synth_mu = float(np.mean(low_ratios_synth))
        mid_synth_mu = float(np.mean(mid_ratios_synth))

        print(f"{p['name']} -> FID: {fid_val}, ACF_mean: {acf_mean}, MMD: {mmd_val}, Energy: {energy_val}", flush=True)

        results.append({
            'dataset': p['name'],
            'FID': fid_val,
            'ACF_mean': acf_mean,
            'ACF_std': acf_std,
            'MMD': mmd_val,
            'Energy': energy_val,
            'SE_real': se_real_mu,
            'SE_synth': se_synth_mu,
            'KS_stat': float(ks_stat),
            'KS_p': float(ks_p),
            'mean_real': mean_real,
            'mean_synth': mean_synth,
            'std_real': std_real,
            'std_synth': std_synth,
            'skew_real': skew_real,
            'skew_synth': skew_synth,
            'kurt_real': kurt_real,
            'kurt_synth': kurt_synth,
            'median_real': median_real,
            'median_synth': median_synth,
            'iqr_real': iqr_real,
            'iqr_synth': iqr_synth,
            'low_real': low_real_mu,
            'mid_real': mid_real_mu,
            'low_synth': low_synth_mu,
            'mid_synth': mid_synth_mu
        })

    except Exception as e:
        print('error processing', p['name'], e, flush=True)
        traceback.print_exc()

# Save combined CSV (use all available keys)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
if len(results) > 0:
    keys = list(results[0].keys())
else:
    keys = ['dataset']
with open(output_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(keys)
    for r in results:
        w.writerow([r.get(k, '') for k in keys])

print('Saved combined CSV to', output_csv, flush=True)

# Print contents
with open(output_csv, 'r') as f:
    print(f.read(), flush=True)
