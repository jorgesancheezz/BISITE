import os
import numpy as np
from scipy import linalg
from scipy import stats
from scipy.stats import kurtosis, skew
import csv
import matplotlib.pyplot as plt

# Load the .npy files
def load_npy(file_path):
    try:
        return np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Calculate statistics for a dataset
def calculate_statistics_numeric(data):
    if data is None:
        return None
    data_flat = data.flatten()  # Flatten to 1D
    return {
        "mean": float(np.mean(data_flat)),
        "std": float(np.std(data_flat)),
        "min": float(np.min(data_flat)),
        "max": float(np.max(data_flat)),
        "median": float(np.median(data_flat)),
        "kurtosis": float(kurtosis(data_flat)),
        "skewness": float(skew(data_flat)),
        "absolute_error": float(np.mean(np.abs(data_flat - np.mean(data_flat)))),
    }


def format_stats_plain(stats_numeric, decimals: int = 6):
    """Formatea en notación decimal fija (sin científica), con 'decimals' decimales."""
    fmt = f"{{:.{decimals}f}}"
    return {k: fmt.format(v) for k, v in stats_numeric.items()}

# Compare statistics between multiple datasets
def compare_statistics(*stats_list):
    if any(s is None for s in stats_list):
        print("One of the datasets could not be loaded.")
        return

    print("Comparison of Statistics:")
    # Admitimos tanto dicts numéricos como ya formateados
    keys = list(stats_list[0].keys())
    for key in keys:
        print(f"{key.capitalize()}:")
        for i, s in enumerate(stats_list, start=1):
            print(f"  Dataset {i}: {s[key]}")
        print()

def save_statistics_to_csv(output_file, labeled_stats):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Statistic"] + [f"Dataset: {lbl}" for lbl, _ in labeled_stats]
        writer.writerow(header)
        keys = list(labeled_stats[0][1].keys())
        for key in keys:
            row = [key.capitalize()]
            for _, st in labeled_stats:
                row.append(st[key])
            writer.writerow(row)

def plot_statistics_separate(output_dir, labeled_numeric_stats):
    os.makedirs(output_dir, exist_ok=True)
    labels = [lbl for lbl, _ in labeled_numeric_stats]
    keys = list(labeled_numeric_stats[0][1].keys())
    for key in keys:
        values = [float(stats[key]) for _, stats in labeled_numeric_stats]
        x = np.arange(len(labels))
        plt.figure(figsize=(12, 7))
        bars = plt.bar(x, values, edgecolor='black', linewidth=1.5, color=['#ffcccb', '#ffebcd', '#e6e6fa', '#d8bfd8', '#add8e6', '#f5deb3'][:len(labels)])
        plt.title(f"{key.capitalize()} por dataset")
        plt.ylabel("Valor")
        plt.xticks(x, labels, rotation=30, ha='right')
        # Anotaciones en notación decimal fija
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.4f}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"stat_{key}.png")
        plt.savefig(out_path)
        plt.close()


# -------- FID y otras métricas pareadas (vs referencia) -------- #
def compute_spectrogram_features(data: np.ndarray, nperseg: int = 256, noverlap: int = 128) -> np.ndarray:
    from scipy.signal import spectrogram

    if data.ndim != 3 or data.shape[-1] != 1:
        raise ValueError(f"Expected (N, T, 1), got {data.shape}")
    feats = []
    for i in range(data.shape[0]):
        _, _, Sxx = spectrogram(data[i, :, 0], nperseg=nperseg, noverlap=noverlap)
        v = np.log1p(Sxx).mean(axis=1)
        feats.append(v.astype(np.float32))
    return np.stack(feats, axis=0)


# ---------- Helpers de visualización PSD ---------- #
def _smooth_db_curve(db: np.ndarray, frac: float = 0.03, min_window: int = 7, polyorder: int = 2) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return db
    n = len(db)
    if n < min_window:
        return db
    win = max(min_window, int(n * frac))
    # La ventana debe ser impar y <= n
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n - 1 if (n - 1) % 2 == 1 else n - 2
        if win < min_window:
            return db
    try:
        return savgol_filter(db, win, polyorder)
    except Exception:
        return db


def compute_mu_sigma(feats: np.ndarray):
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    def _sqrtm_compat(A: np.ndarray) -> np.ndarray:
        res = linalg.sqrtm(A)
        return res[0] if isinstance(res, tuple) else res

    diff = mu1 - mu2
    covmean = _sqrtm_compat(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _sqrtm_compat((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    return fid


def flatten_sample(data: np.ndarray, max_points: int = 200_000, seed: int = 42) -> np.ndarray:
    v = data.reshape(-1)
    if v.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(v.size, size=max_points, replace=False)
        v = v[idx]
    return v.astype(np.float32)


def compute_pair_metrics(ref: np.ndarray, cmp: np.ndarray) -> dict:
    # FID en espacio de features de espectrograma
    feats_ref = compute_spectrogram_features(ref)
    feats_cmp = compute_spectrogram_features(cmp)
    mu1, s1 = compute_mu_sigma(feats_ref)
    mu2, s2 = compute_mu_sigma(feats_cmp)
    fid = frechet_distance(mu1, s1, mu2, s2)

    # Distancia de energía y KS entre distribuciones de valores
    a = flatten_sample(ref)
    b = flatten_sample(cmp)
    try:
        energy = float(stats.energy_distance(a, b))
    except Exception:
        energy = float('nan')
    ks = stats.ks_2samp(a, b, method='auto')
    return {
        'FID': fid,
        'energy_distance': energy,
        'ks_stat': float(ks.statistic),
        'ks_pvalue': float(ks.pvalue),
    }

def highpass_filter_dataset(data: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if data is None:
        return None
    if data.ndim != 3 or data.shape[-1] != 1:
        return data
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, cutoff_hz, btype='highpass', fs=fs, output='sos')
    out = np.empty_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        x = data[i, :, 0]
        y = sosfiltfilt(sos, x).astype(np.float32)
        out[i, :, 0] = y
    return out

def save_pair_metrics_csv(path: str, ref_label: str, labeled_datas: list):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['reference', 'dataset', 'FID', 'energy_distance', 'ks_stat', 'ks_pvalue'])
        ref_data = None
        for lbl, d in labeled_datas:
            if lbl == ref_label:
                ref_data = d
                break
        if ref_data is None:
            raise ValueError(f"Reference dataset '{ref_label}' not found among { [l for l,_ in labeled_datas] }")
        for lbl, d in labeled_datas:
            if lbl == ref_label:
                continue
            m = compute_pair_metrics(ref_data, d)
            w.writerow([ref_label, lbl, f"{m['FID']:.6f}", f"{m['energy_distance']:.6f}", f"{m['ks_stat']:.6f}", f"{m['ks_pvalue']:.6f}"])

def plot_bar_metric(path: str, title: str, metric_name: str, ref_label: str, labeled_datas: list):
    values = []
    labels = []
    # encontrar ref
    ref_data = None
    for lbl, d in labeled_datas:
        if lbl == ref_label:
            ref_data = d
            break
    if ref_data is None:
        return
    for lbl, d in labeled_datas:
        if lbl == ref_label:
            continue
        m = compute_pair_metrics(ref_data, d)
        values.append(m[metric_name])
        labels.append(lbl)
    x = np.arange(len(labels))
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x, values, edgecolor='black', linewidth=1.5, color=['#ffcccb', '#ffebcd', '#e6e6fa', '#d8bfd8', '#add8e6', '#f5deb3'][:len(labels)])
    plt.title(title)
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel(metric_name)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.4f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_average_spectrum(path: str, labeled_datas: list, nperseg: int = 1024, noverlap: int = 512, fs: float = 1.0, xlim: tuple | None = None, ylim: tuple | None = None, smooth: bool = True, smooth_frac: float = 0.03, vlines_hz: list | None = None, title: str | None = None):

    from scipy.signal import welch
    import matplotlib.pyplot as _plt

    _plt.figure(figsize=(13.5, 7.5), dpi=180)
    for lbl, d in labeled_datas:
        if d.ndim != 3 or d.shape[-1] != 1:
            continue
        ps_list = []
        f = None
        for i in range(d.shape[0]):
            f, Pxx = welch(d[i, :, 0], fs=fs, nperseg=nperseg, noverlap=noverlap)
            ps_list.append(Pxx)
        if not ps_list:
            continue
        ps_mean = np.mean(np.stack(ps_list, axis=0), axis=0)
        db = 10*np.log10(ps_mean + 1e-12)
        if smooth:
            db = _smooth_db_curve(db, frac=smooth_frac)
        _plt.plot(f, db, label=lbl, linewidth=2.2)  # dB/Hz
    _plt.xlabel('Frecuencia (Hz)' if fs != 1.0 else 'Frecuencia (bins normalizados)')
    _plt.ylabel('Potencia (dB/Hz)')
    _plt.title(title or 'Espectro de potencia promedio por dataset')
    _plt.grid(True, which='both', linestyle='--', alpha=0.4)
    if xlim is not None:
        _plt.xlim(xlim)
    if ylim is not None:
        _plt.ylim(ylim)
    if vlines_hz is not None:
        for v in vlines_hz:
            if v is not None and (xlim is None or (v >= xlim[0] and v <= xlim[1])):
                _plt.axvline(v, color='gray', linestyle=':', alpha=0.5, linewidth=1.2)
    _plt.legend()
    _plt.tight_layout()
    _plt.savefig(path, dpi=180)
    _plt.close()

def plot_pair_spectra_vs_ref(output_dir: str, ref_label: str, labeled_datas: list, nperseg: int = 1024, noverlap: int = 512, max_samples: int | None = None, fs: float = 1.0, xlim: tuple | None = None, ylim: tuple | None = None, smooth: bool = True, smooth_frac: float = 0.03, vlines_hz: list | None = None):

    from scipy.signal import welch

    os.makedirs(output_dir, exist_ok=True)

    # Localiza referencia
    ref_data = None
    for lbl, d in labeled_datas:
        if lbl == ref_label:
            ref_data = d
            break
    if ref_data is None:
        return

    def avg_psd(d: np.ndarray):
        if d is None or d.ndim != 3 or d.shape[-1] != 1:
            return None, None
        N = d.shape[0]
        indices = np.arange(N)
        if max_samples is not None and N > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(N, size=max_samples, replace=False)
        ps_list = []
        f = None
        for i in indices:
            f, Pxx = welch(d[i, :, 0], fs=fs, nperseg=nperseg, noverlap=noverlap)
            ps_list.append(Pxx)
        if not ps_list:
            return None, None
        ps_mean = np.mean(np.stack(ps_list, axis=0), axis=0)
        return f, ps_mean

    # PSD medio de la referencia (una sola vez)
    f_ref, ps_ref = avg_psd(ref_data)
    if f_ref is None or ps_ref is None:
        return

    def _safe(name: str) -> str:
        return name.replace('.npy', '').replace(' ', '_').replace('/', '_')

    ref_short = _safe(ref_label)

    # Precalcular dB de la referencia
    ref_db = 10 * np.log10(ps_ref + 1e-12)
    if smooth:
        ref_db = _smooth_db_curve(ref_db, frac=smooth_frac)

    # Para fijar los mismos límites Y en todas las figuras, calcula min/max globales
    db_min = np.min(ref_db)
    db_max = np.max(ref_db)

    cache = {}
    for lbl, d in labeled_datas:
        if lbl == ref_label:
            continue
        f_cmp, ps_cmp = avg_psd(d)
        if f_cmp is None or ps_cmp is None:
            continue
        db_cmp = 10 * np.log10(ps_cmp + 1e-12)
        if smooth:
            db_cmp = _smooth_db_curve(db_cmp, frac=smooth_frac)
        cache[lbl] = (f_cmp, db_cmp)
        db_min = min(db_min, np.min(db_cmp))
        db_max = max(db_max, np.max(db_cmp))

    # Margen para que no se pegue el trazo a los bordes
    margin = 1.5
    auto_ylim = (db_min - margin, db_max + margin)

    for lbl, (f_cmp, db_cmp) in cache.items():
        plt.figure(figsize=(13.5, 7.5), dpi=180)
        plt.plot(f_ref, ref_db, label=ref_label, color='#1f77b4', linewidth=2)
        plt.plot(f_cmp, db_cmp, label=lbl, color='#d62728', linewidth=2, alpha=0.9)
        plt.xlabel('Frecuencia (Hz)' if fs != 1.0 else 'Frecuencia (bins normalizados)')
        plt.ylabel('Potencia (dB/Hz)')
        plt.title(f'PSD medio: {lbl} vs {ref_label}')
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        # Mismos límites Y para facilitar comparación visual
        plt.ylim(ylim if ylim is not None else auto_ylim)
        # Ticks X “bonitos” si está en Hz
        if fs != 1.0:
            import math
            xmax = fs / 2.0
            step = 20 if fs >= 200 else max(10, math.floor(fs/10))
            xticks = np.arange(0, xmax + 1e-6, step)
            plt.xticks(xticks)
        if xlim is not None:
            plt.xlim(xlim)
        if vlines_hz is not None:
            for v in vlines_hz:
                if v is not None and (xlim is None or (v >= xlim[0] and v <= xlim[1])):
                    plt.axvline(v, color='gray', linestyle=':', alpha=0.5, linewidth=1.2)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"psd_{_safe(lbl)}_vs_{ref_short}.png")
        plt.savefig(out_path, dpi=180)
        plt.close()


def plot_single_spectra(output_dir: str, labeled_datas: list, nperseg: int = 1024, noverlap: int = 512, max_samples: int | None = None, fs: float = 1.0, xlim: tuple | None = None, smooth: bool = True, smooth_frac: float = 0.03, vlines_hz: list | None = None):
    from scipy.signal import welch

    os.makedirs(output_dir, exist_ok=True)

    def avg_psd(d: np.ndarray):
        if d is None or d.ndim != 3 or d.shape[-1] != 1:
            return None, None
        N = d.shape[0]
        indices = np.arange(N)
        if max_samples is not None and N > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(N, size=max_samples, replace=False)
        ps_list = []
        f = None
        for i in indices:
            f, Pxx = welch(d[i, :, 0], fs=fs, nperseg=nperseg, noverlap=noverlap)
            ps_list.append(Pxx)
        if not ps_list:
            return None, None
        ps_mean = np.mean(np.stack(ps_list, axis=0), axis=0)
        return f, ps_mean

    cache = {}
    db_min, db_max = None, None
    for lbl, d in labeled_datas:
        f, ps = avg_psd(d)
        if f is None or ps is None:
            continue
        db = 10 * np.log10(ps + 1e-12)
        cache[lbl] = (f, db)
        mn, mx = float(np.min(db)), float(np.max(db))
        db_min = mn if db_min is None else min(db_min, mn)
        db_max = mx if db_max is None else max(db_max, mx)

    if not cache:
        return

    margin = 1.5
    ylim = (db_min - margin, db_max + margin)

    def _safe(name: str) -> str:
        return name.replace('.npy', '').replace(' ', '_').replace('/', '_')

    for lbl, (f, db) in cache.items():
        if smooth:
            db = _smooth_db_curve(db, frac=smooth_frac)
        plt.figure(figsize=(13.5, 7.5), dpi=180)
        plt.plot(f, db, color='#1f77b4', linewidth=2.2)
        plt.xlabel('Frecuencia (Hz)' if fs != 1.0 else 'Frecuencia (bins normalizados)')
        plt.ylabel('Potencia (dB/Hz)')
        plt.title(f'PSD medio: {lbl}')
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.ylim(ylim)
        if fs != 1.0:
            import math
            xmax = fs / 2.0
            step = 20 if fs >= 200 else max(10, math.floor(fs/10))
            xticks = np.arange(0, xmax + 1e-6, step)
            plt.xticks(xticks)
        if xlim is not None:
            plt.xlim(xlim)
        if vlines_hz is not None:
            for v in vlines_hz:
                if v is not None and (xlim is None or (v >= xlim[0] and v <= xlim[1])):
                    plt.axvline(v, color='gray', linestyle=':', alpha=0.5, linewidth=1.2)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"psd_{_safe(lbl)}.png")
        plt.savefig(out_path, dpi=180)
        plt.close()

if __name__ == "__main__":
    # Frecuencia de muestreo para los PSD en Hz (ajústala si tus señales no son de 250 Hz)
    FS_HZ = 250.0
    HF_CUTOFF_HZ = 40.0  # umbral de alta frecuencia para estadísticas filtradas
    # Lista de ficheros a comparar (puedes añadir/quitar aquí)
    files = [
        "PULSOVITAL/results/003.npy",
        "PULSOVITAL/results/sintetico.npy",
        "PULSOVITAL/results/004.npy",
        "PULSOVITAL/results/005.npy",
        "PULSOVITAL/results/006.npy",
        "PULSOVITAL/results/007.npy",
    ]
    # Filtrar solo los que existan
    files = [f for f in files if os.path.exists(f)]
    labels = [os.path.basename(f) for f in files]

    # Cargar
    datas = [load_npy(f) for f in files]
    # Calcular estadísticas y filtrar (ahora incluimos mean; seguimos omitiendo min/max/median)
    keep_keys = ["mean", "std", "kurtosis", "skewness", "absolute_error"]
    stats_numeric_all = [calculate_statistics_numeric(d) for d in datas]
    stats_numeric = [{k: s[k] for k in keep_keys if k in s} for s in stats_numeric_all]
    stats_formatted = [format_stats_plain(s, decimals=6) for s in stats_numeric]

    # Comparar (imprime por consola)
    compare_statistics(*stats_formatted)

    # Guardar CSV con todas las columnas
    output_csv = "PULSOVITAL/results/comparison_statistics.csv"
    save_statistics_to_csv(output_csv, list(zip(labels, stats_formatted)))
    print(f"Statistics saved to {output_csv}")

    # Generar plots por estadística
    output_dir = "PULSOVITAL/results/comparison_plots"
    plot_statistics_separate(output_dir, list(zip(labels, stats_numeric)))
    print(f"Per-statistic plots saved to {output_dir}")

    # Métricas pareadas vs referencia
    # Preferimos comparar contra el sintético si existe
    preferred_refs = ["sintetico.npy", "003.npy"]
    ref_label = None
    for cand in preferred_refs:
        if cand in labels:
            ref_label = cand
            break
    if ref_label is None:
        ref_label = labels[0]
    pair_csv = "PULSOVITAL/results/pair_metrics_vs_ref.csv"
    save_pair_metrics_csv(pair_csv, ref_label, list(zip(labels, datas)))
    print(f"Pairwise metrics vs {ref_label} saved to {pair_csv}")

    # Plots de FID y energy distance vs referencia (mantenemos)
    plot_bar_metric(os.path.join(output_dir, "fid_vs_ref.png"), f"FID vs {ref_label}", "FID", ref_label, list(zip(labels, datas)))
    plot_bar_metric(os.path.join(output_dir, "energy_distance_vs_ref.png"), f"Energy distance vs {ref_label}", "energy_distance", ref_label, list(zip(labels, datas)))
    # PSD de banda completa: juntos (overlay) y comparados vs sintético
    # Juntos (overlay)
    avg_full_path = os.path.join(output_dir, "avg_spectrum_full.png")
    plot_average_spectrum(
        avg_full_path,
        list(zip(labels, datas)),
        fs=FS_HZ,
        xlim=None,
        ylim=None,
        smooth=True,
        smooth_frac=0.03,
        vlines_hz=[50.0, 60.0],
        title='Espectro de potencia promedio (completo)'
    )
    print(f"Full-band overlay PSD saved to {avg_full_path}")

    # Comparados vs referencia (preferimos sintetico.npy si existe)
    pair_full_dir = os.path.join(output_dir, "psd_vs_ref_full")
    plot_pair_spectra_vs_ref(
        pair_full_dir,
        ref_label,
        list(zip(labels, datas)),
        fs=FS_HZ,
        xlim=None,
        ylim=None,
        smooth=True,
        smooth_frac=0.03,
        vlines_hz=[50.0, 60.0]
    )
    print(f"Full-band pairwise PSDs vs {ref_label} saved to {pair_full_dir}")
