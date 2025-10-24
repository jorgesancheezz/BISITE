import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent
FILES = [

    RESULTS_DIR / 'resultados_ecg_nsr_autocorr_stats_exclude_lag0.csv',
    RESULTS_DIR / 'resultados_ecg_af_autocorr_stats.csv',
]


def load_and_normalize():
    dfs = []
    for p in FILES:
        if p.exists():
            df = pd.read_csv(p)
            name = p.name.lower()
            if 'nsr' in name:
                ritmo = 'NSR'
            else:
                ritmo = 'AF'
            df['ritmo'] = ritmo
            dfs.append(df)
    if not dfs:
        raise SystemExit('No CSVs found')
    data = pd.concat(dfs, ignore_index=True)
    col_map = {}
    for c in data.columns:
        lc = c.lower()
        if 'orig' in lc and 'skew' in lc:
            col_map[c] = 'orig_skewness'
        if 'orig' in lc and 'kurt' in lc:
            col_map[c] = 'orig_kurtosis'
        if 'acorr' in lc and 'skew' in lc:
            col_map[c] = 'acorr_skewness_no_lag0'
        if 'acorr' in lc and 'kurt' in lc:
            col_map[c] = 'acorr_kurtosis_no_lag0'
    data = data.rename(columns=col_map)
    data = data[data['ritmo'].isin(['AF', 'NSR'])]
    return data


def sample_vals(vals_full, fraction, rng):
    vals_full = vals_full[np.isfinite(vals_full)]
    n = vals_full.size
    if n == 0:
        return vals_full
    if fraction >= 1.0:
        return vals_full
    k = max(1, int(round(n * fraction)))
    if k >= n:
        return vals_full
    idx = rng.choice(n, size=k, replace=False)
    return vals_full[idx]


def plot_skewness_forced(data, forced_xlim=(-10.0, 10.0), fraction=1.0, seed=42):
    rng = np.random.RandomState(seed)
    metric = 'orig_skewness'
    rhythms = ['AF', 'NSR']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plt.suptitle(f'Normality checks (forced x-limits {forced_xlim}): {metric} (fraction={fraction})')
    for i, ritmo in enumerate(rhythms):
        vals_full = data.loc[data['ritmo'] == ritmo, metric].dropna().values
        vals = sample_vals(vals_full, fraction, rng)
        ax_hist = axes[i, 0]
        ax_qq = axes[i, 1]
        if vals.size == 0:
            ax_hist.text(0.5, 0.5, 'No data', ha='center')
            ax_qq.text(0.5, 0.5, 'No data', ha='center')
            continue
        ax_hist.set_title(f'{ritmo} (n={vals.size})')
        ax_hist.set_xlabel('Orig Skewness')
        ax_hist.set_ylabel('Density')
        # Use forced x-limits: filter values to plotting window to avoid passing 'range' to seaborn
        rmin, rmax = float(forced_xlim[0]), float(forced_xlim[1])
        vals_clip = vals[(vals >= rmin) & (vals <= rmax)]
        if vals_clip.size == 0:
            # if nothing in the forced window, plot full histogram but keep xlim forced
            sns.histplot(vals, bins=80, kde=False, ax=ax_hist, stat='density', color='tab:blue', element='step', alpha=0.6)
            try:
                xs = np.linspace(rmin, rmax, 400)
                kde = stats.gaussian_kde(vals)
                ys = kde(xs)
                ax_hist.plot(xs, ys, color='C1', linewidth=1.5)
            except Exception:
                sns.kdeplot(vals, ax=ax_hist, color='C1')
        else:
            sns.histplot(vals_clip, bins=80, kde=False, ax=ax_hist, stat='density', color='tab:blue', element='step', alpha=0.6)
            try:
                xs = np.linspace(rmin, rmax, 400)
                kde = stats.gaussian_kde(vals_clip)
                ys = kde(xs)
                ax_hist.plot(xs, ys, color='C1', linewidth=1.5)
            except Exception:
                sns.kdeplot(vals_clip, ax=ax_hist, color='C1')
        ax_hist.set_xlim(rmin, rmax)
        ax_hist.text(0.98, 0.95, f'Forced x-limits: {rmin}..{rmax}', transform=ax_hist.transAxes, ha='right', va='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        # QQ-plot
        try:
            stats.probplot(vals, dist='norm', plot=ax_qq)
            ax_qq.set_title(f'QQ-plot {ritmo}')
        except Exception:
            q = np.linspace(0.01, 0.99, 100)
            theo = stats.norm.ppf(q)
            emp = np.quantile(vals, q)
            ax_qq.plot(theo, emp, marker='o', linestyle='none')
            ax_qq.plot(theo, theo, color='k', linestyle='--')
            ax_qq.set_title(f'QQ-plot {ritmo} (fallback)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outfile = RESULTS_DIR / f'normality_orig_skewness_forced_{int(fraction*100)}.png'
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', '-f', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data = load_and_normalize()
    out = plot_skewness_forced(data, forced_xlim=(-10.0, 10.0), fraction=args.fraction, seed=args.seed)
    print('Saved', out)


if __name__ == '__main__':
    main()
