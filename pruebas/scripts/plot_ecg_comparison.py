"""Create side-by-side comparison plots between article and 1024seq for AF and NSR.
Saves PNGs to `notebooks/outputs/` and opens them with the default image viewer on Windows.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch


def load(p: Path):
    a = np.load(p)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a.reshape(a.shape[0], a.shape[1])
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a


def mean_std_plot(ax, A, fs=300.0, label=None):
    mean = np.mean(A, axis=0)
    std = np.std(A, axis=0)
    x = np.arange(len(mean)) / fs
    ax.plot(x, mean, label=label)
    ax.fill_between(x, mean-std, mean+std, alpha=0.25)


def psd_mean(A, fs=300.0):
    ps = []
    for s in A[:min(256, len(A))]:
        f,P = welch(s, fs=fs, nperseg=min(1024, len(s)))
        ps.append(P)
    if not ps:
        return None, None
    return f, np.mean(ps, axis=0)


def compare_pair(p1024: Path, part: Path, cls: str, out_dir: Path):
    A = load(p1024)
    B = load(part)
    fs = 300.0
    T = min(A.shape[1], B.shape[1])
    A = A[:, :T]
    B = B[:, :T]

    fig, axes = plt.subplots(2,2, figsize=(12,8))
    ax0 = axes[0,0]
    mean_std_plot(ax0, A, fs=fs, label='1024seq')
    mean_std_plot(ax0, B, fs=fs, label='Article')
    ax0.set_title(f'{cls} — Mean ± std')
    ax0.set_xlabel('time (s)')
    ax0.legend()

    ax1 = axes[0,1]
    # sample traces
    n = min(6, A.shape[0], B.shape[0])
    for i in range(n):
        ax1.plot(np.arange(T)/fs, A[i] + i*0.0, color='C0', alpha=0.6)
    for i in range(n):
        ax1.plot(np.arange(T)/fs, B[i] + i*0.0, color='C1', alpha=0.4)
    ax1.set_title('Example traces (overlay)')

    ax2 = axes[1,0]
    fA, pA = psd_mean(A, fs=fs)
    fB, pB = psd_mean(B, fs=fs)
    if pA is not None:
        ax2.semilogy(fA, pA, label='1024seq')
    if pB is not None:
        ax2.semilogy(fB, pB, label='Article')
    ax2.set_title('Mean PSD')
    ax2.set_xlabel('Hz')
    ax2.legend()

    ax3 = axes[1,1]
    # histogram of means
    ax3.hist(A.mean(axis=1), bins=50, alpha=0.6, label='1024seq')
    ax3.hist(B.mean(axis=1), bins=50, alpha=0.6, label='Article')
    ax3.set_title('Distribution of sample means')
    ax3.legend()

    fig.tight_layout()
    outp = out_dir / f'ecg_comparison_{cls}.png'
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    print('Saved', outp)
    return outp


def main():
    out_dir = Path('notebooks/outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        (Path('1024seq_AF.npy'), Path('sssd_article_AF.npy'), 'AF'),
        (Path('1024seq_NSR.npy'), Path('sssd_article_NSR.npy'), 'NSR'),
    ]
    saved = []
    for p1024, part, cls in pairs:
        if not p1024.exists() or not part.exists():
            print('Missing pair', p1024, part)
            continue
        saved.append(compare_pair(p1024, part, cls, out_dir))

    # open images on Windows
    import subprocess, sys
    for s in saved:
        try:
            if sys.platform.startswith('win'):
                subprocess.run(['cmd', '/c', 'start', '', str(s)], check=False)
            else:
                subprocess.run(['xdg-open', str(s)], check=False)
        except Exception:
            pass


if __name__ == '__main__':
    main()
