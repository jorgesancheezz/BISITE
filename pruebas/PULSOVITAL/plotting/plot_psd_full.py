import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_npy(path: str):
    try:
        data = np.load(path)
        # Handle .npz files by extracting the first array
        if isinstance(data, np.lib.npyio.NpzFile):
            keys = list(data.keys())
            if keys:
                return data[keys[0]]
            else:
                print(f"No arrays found in {path}")
                return None
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def _maybe_smooth(y: np.ndarray, frac: float = 0.03, min_points: int = 7, polyorder: int = 2) -> np.ndarray:
    """Optionally smooth a 1D curve with Savitzky–Golay if available and long enough."""
    try:
        from scipy.signal import savgol_filter  # type: ignore
    except Exception:
        return y
    n = len(y)
    if n < min_points:
        return y
    # Compute an odd window length proportional to the signal length
    win = max(min_points, int(n * frac))
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if (n - 1) % 2 == 1 else max(min_points, n - 2)
    if win < min_points or win <= 1:
        return y
    try:
        return savgol_filter(y, win, polyorder)
    except Exception:
        return y


def _avg_psd(data: np.ndarray, fs: float, nperseg: int, noverlap: int):
    """Average Welch PSD across samples of shape (N, T, 1). Returns (f, ps_mean) or (None, None)."""
    if data is None or data.ndim != 3 or data.shape[-1] != 1:
        return None, None
    from scipy.signal import welch
    ps_list = []
    f = None
    for i in range(data.shape[0]):
        f, Pxx = welch(data[i, :, 0], fs=fs, nperseg=nperseg, noverlap=noverlap)
        ps_list.append(Pxx)
    if not ps_list:
        return None, None
    return f, np.mean(np.stack(ps_list, axis=0), axis=0)


def _to_db(ps: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(ps + 1e-12)


def _safe_name(name: str) -> str:
    return name.replace('.npy', '').replace(' ', '_').replace('/', '_')


def plot_average_spectrum(
    output_path: str,
    labeled_datas: list,
    *,
    fs: float = 250.0,
    nperseg: int = 1024,
    noverlap: int = 512,
    smooth: bool = True,
    vlines_hz: list = None,
    title: str | None = None,
):
    plt.figure(figsize=(13.5, 7.5), dpi=180)
    for lbl, d in labeled_datas:
        f, ps = _avg_psd(d, fs, nperseg, noverlap)
        if f is None:
            continue
        db = _to_db(ps)
        if smooth:
            db = _maybe_smooth(db)
        plt.plot(f, db, label=lbl, linewidth=2.2)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Potencia (dB/Hz)')
    plt.title(title or 'Espectro de potencia promedio (completo)')
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    if vlines_hz:
        for v in vlines_hz:
            plt.axvline(v, color='gray', linestyle=':', alpha=0.5, linewidth=1.2)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_pair_spectra_vs_ref(
    output_dir: str,
    ref_label: str,
    labeled_datas: list,
    *,
    fs: float = 250.0,
    nperseg: int = 1024,
    noverlap: int = 512,
    smooth: bool = True,
    vlines_hz: list = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Find reference
    ref_data = None
    for lbl, d in labeled_datas:
        if lbl == ref_label:
            ref_data = d
            break
    if ref_data is None:
        raise ValueError(f"Reference '{ref_label}' not found")

    f_ref, ps_ref = _avg_psd(ref_data, fs, nperseg, noverlap)
    if f_ref is None:
        return
    ref_db = _to_db(ps_ref)
    if smooth:
        ref_db = _maybe_smooth(ref_db)

    for lbl, d in labeled_datas:
        if lbl == ref_label:
            continue
        f_cmp, ps_cmp = _avg_psd(d, fs, nperseg, noverlap)
        if f_cmp is None:
            continue
        db_cmp = _to_db(ps_cmp)
        if smooth:
            db_cmp = _maybe_smooth(db_cmp)

        plt.figure(figsize=(13.5, 7.5), dpi=180)
        plt.plot(f_ref, ref_db, label=ref_label, linewidth=2)
        plt.plot(f_cmp, db_cmp, label=lbl, linewidth=2, alpha=0.9)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Potencia (dB/Hz)')
        plt.title(f'PSD medio (completo): {lbl} vs {ref_label}')
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        if vlines_hz:
            for v in vlines_hz:
                plt.axvline(v, color='gray', linestyle=':', alpha=0.5, linewidth=1.2)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"psd_{_safe_name(lbl)}_vs_{_safe_name(ref_label)}.png")
        plt.savefig(out_path, dpi=180)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot full-band PSD average and pairwise vs reference")
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--out-dir", type=str, default=os.path.join("PULSOVITAL", "results", "comparison_plots"))
    ap.add_argument(
        "--files",
        nargs="*",
        default=[
            os.path.join("PULSOVITAL", "results", "003.npy"),
            os.path.join("PULSOVITAL", "results", "004.npy"),
            os.path.join("PULSOVITAL", "results", "005.npy"),
            os.path.join("PULSOVITAL", "results", "006.npy"),
            os.path.join("PULSOVITAL", "results", "007.npy"),
            os.path.join("PULSOVITAL", "results", "synthetic_scale_0.1270_noise_0.05.npz"),
            os.path.join("PULSOVITAL", "results", "synthetic_scale_0.1290_noise_0.1.npz"),
        ],
    )
    args = ap.parse_args()

    files = [f for f in args.files if os.path.exists(f)]
    if not files:
        raise SystemExit("No input files found")
    labels = [os.path.basename(f) for f in files]
    datas = [load_npy(f) for f in files]

    # Prefer sintetico as reference if present
    ref_label = next((lbl for lbl in labels if lbl == "sintetico.npy"), labels[0])

    avg_path = os.path.join(args.out_dir, "avg_spectrum_full.png")
    plot_average_spectrum(avg_path, list(zip(labels, datas)), fs=args.fs, vlines_hz=[50.0, 60.0])

    pair_dir = os.path.join(args.out_dir, "psd_vs_ref_full")
    plot_pair_spectra_vs_ref(pair_dir, ref_label, list(zip(labels, datas)), fs=args.fs, vlines_hz=[50.0, 60.0])
    print(f"Saved: {avg_path} and folder {pair_dir}")


if __name__ == "__main__":
    main()
