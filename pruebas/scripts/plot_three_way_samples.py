"""Plot 3-way comparison (synthetic / real / article) showing 5 traces each for AF and NSR.

Defaults point to known repo locations; saves PNGs to `notebooks/outputs` and opens them.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random


# Prefer synthetic files from PULSOVITAL/npy_output if available, otherwise fall back
def _default_pairs():
    base_syn = Path('PULSOVITAL/npy_output')
    pairs = {}
    af_syn = base_syn / 'AF_processed_1024x3000x1.npy'
    nsr_syn = base_syn / 'NSR_processed_1024x3000x1.npy'
    pairs['AF'] = {
        'synthetic': af_syn if af_syn.exists() else Path('compare_out_generated/sssd_pulso_run1/sssd_AF_1024.npy'),
        'real': Path('1024seq_AF.npy'),
        'article': Path('sssd_article_AF.npy'),
    }
    pairs['NSR'] = {
        'synthetic': nsr_syn if nsr_syn.exists() else Path('compare_out_generated/sssd_pulso_run1/sssd_NSR_1024.npy'),
        'real': Path('1024seq_NSR.npy'),
        'article': Path('sssd_article_NSR.npy'),
    }
    return pairs

PAIRS = _default_pairs()


def load_arr(p: Path):
    a = np.load(p)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a.reshape(a.shape[0], a.shape[1])
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a


def plot_three_way(cls, paths, out_dir: Path, n_samples=5, fs=300.0):
    data = {}
    for k,v in paths.items():
        if not v.exists():
            raise FileNotFoundError(f'Missing file for {k}: {v}')
        data[k] = load_arr(v)

    # ensure same length T
    T = min(data['synthetic'].shape[1], data['real'].shape[1], data['article'].shape[1])
    for k in data: data[k] = data[k][:, :T]

    fig, axs = plt.subplots(1, 3, figsize=(15,4), sharey=True)
    titles = ['Synthetic', 'Real (1024seq)', 'Article']
    keys = ['synthetic','real','article']

    rng = random.Random(42)

    for ax, title, key in zip(axs, titles, keys):
        A = data[key]
        N = A.shape[0]
        inds = list(range(N))
        if N > n_samples:
            rng.shuffle(inds)
            inds = inds[:n_samples]
        else:
            inds = inds[:n_samples]
        for i, idx in enumerate(inds):
            ax.plot(A[idx], alpha=0.8, label=f's{i}' if i<1 else None)
        ax.set_title(title)
        ax.set_xlabel('time (samples)')
    axs[0].set_ylabel('amplitude')
    fig.suptitle(f'{cls} — 5 traces each (synthetic / real / article)')
    fig.tight_layout(rect=[0,0,1,0.96])

    outp = out_dir / f'3way_{cls}.png'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    return outp


def main():
    out_dir = Path('notebooks/outputs')
    saved = []
    missing = []
    for cls,paths in PAIRS.items():
        try:
            p = plot_three_way(cls, paths, out_dir, n_samples=5)
            saved.append(p)
            print('Saved', p)
        except FileNotFoundError as e:
            missing.append(str(e))
            print('Skipping', cls, '-', e)

    # open saved images on Windows
    import subprocess, sys
    for s in saved:
        try:
            if sys.platform.startswith('win'):
                subprocess.run(['cmd','/c','start','', str(s)], check=False)
            else:
                subprocess.run(['xdg-open', str(s)], check=False)
        except Exception:
            pass

    if missing:
        print('Missing files:', '\n'.join(missing))


if __name__ == '__main__':
    main()
