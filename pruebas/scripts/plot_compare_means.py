#!/usr/bin/env python3
"""Plot and save comparison of per-sample mean distributions

Loads `PULSOVITAL/Metricas/sssd_article_AF_proc_proc_3000.npy` and
`1024seq_AF_normalized.npy`, computes per-sample means and saves an overlaid
histogram + KDE to `notebooks/outputs/mean_distribution_comparison.png`.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_signals(p):
    a = np.load(p)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim == 1:
        a = a[np.newaxis, :]
    return a


def main():
    article_p = os.path.join('PULSOVITAL', 'Metricas', 'sssd_article_AF_proc_proc_3000.npy')
    seq_p = '1024seq_AF_normalized.npy'
    out_dir = os.path.join('notebooks', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(article_p):
        print('Missing', article_p); return 1
    if not os.path.exists(seq_p):
        print('Missing', seq_p); return 1

    art = load_signals(article_p)
    seq = load_signals(seq_p)

    # compute per-sample means
    art_means = art.mean(axis=1)
    seq_means = seq.mean(axis=1)

    plt.figure(figsize=(8,5))
    bins = 100
    plt.hist(art_means, bins=bins, density=True, alpha=0.5, label='article_AF')
    plt.hist(seq_means, bins=bins, density=True, alpha=0.5, label='1024seq_AF_normalized')
    # simple KDE using gaussian smoothing via numpy histogram smoothing
    try:
        import scipy.stats as ss
        kde_a = ss.gaussian_kde(art_means)
        kde_s = ss.gaussian_kde(seq_means)
        xs = np.linspace(min(art_means.min(), seq_means.min()), max(art_means.max(), seq_means.max()), 512)
        plt.plot(xs, kde_a(xs), color='C0')
        plt.plot(xs, kde_s(xs), color='C1')
    except Exception:
        pass

    plt.legend()
    plt.xlabel('Per-sample mean')
    plt.ylabel('Density')
    plt.title('Mean distribution: article_AF vs 1024seq_AF_normalized')

    out_path = os.path.join(out_dir, 'mean_distribution_comparison.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print('Saved plot to', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
