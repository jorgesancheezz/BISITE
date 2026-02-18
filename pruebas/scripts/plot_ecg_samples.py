"""Plot representative ECG samples, mean±std and PSD for given .npy files.

Usage: python scripts/plot_ecg_samples.py --files sssd_article_AF.npy sssd_article_NSR.npy
Saves PNGs to notebooks/outputs/ecg_samples_<basename>.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def load_array(p: Path):
    a = np.load(p)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a.reshape(a.shape[0], a.shape[1])
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a


def plot_for_file(p: Path, out_dir: Path, n_traces=6, fs=300.0):
    A = load_array(p)
    N, T = A.shape
    idxs = np.linspace(0, N-1, min(N, n_traces), dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # plot individual traces (offset)
    ax = axes[0]
    offset = np.max(np.abs(A)) * 1.2 if A.size else 1.0
    for i, ii in enumerate(idxs):
        t = A[ii]
        ax.plot(t + i * offset * 0.1, label=f'sample {ii}')
    ax.set_title(f'{p.name} — {len(idxs)} example traces')
    ax.legend(loc='upper right', fontsize='small')

    # plot mean ± std
    ax = axes[1]
    mean = np.mean(A, axis=0)
    std = np.std(A, axis=0)
    x = np.arange(T) / fs
    ax.plot(x, mean, color='C0', label='mean')
    ax.fill_between(x, mean-std, mean+std, color='C0', alpha=0.3, label='±1 std')
    ax.set_title(f'{p.name} — mean ± std (N={N})')
    ax.set_xlabel('time (s)')
    ax.legend()

    # PSD (mean)
    ax = axes[2]
    ps = []
    for s in A[:min(256, len(A))]:
        f, P = welch(s, fs=fs, nperseg=min(1024, len(s)))
        ps.append(P)
    if ps:
        Pm = np.mean(ps, axis=0)
        ax.semilogy(f, Pm)
        ax.set_xlabel('freq (Hz)')
        ax.set_title('Mean PSD (Welch)')

    fig.tight_layout()
    outp = out_dir / f'ecg_samples_{p.stem}.png'
    ensure_dir(outp)
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    print('Saved plot to', outp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--out-dir', default='notebooks/outputs')
    parser.add_argument('--n-traces', type=int, default=6)
    parser.add_argument('--fs', type=float, default=300.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in args.files:
        p = Path(f)
        if not p.exists():
            print('Missing', p)
            continue
        plot_for_file(p, out_dir, n_traces=args.n_traces, fs=args.fs)


if __name__ == '__main__':
    main()
