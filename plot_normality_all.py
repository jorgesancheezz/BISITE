"""Generate the four normality checks in one script without overwriting existing PNGs.

This script reproduces the four per-metric plots (orig_skewness, orig_kurtosis,
acorr_skewness_no_lag0, acorr_kurtosis_no_lag0) and saves them as
`normality_{metric}_combined_fracXX.png` in the same `resultados2` folder.

It intentionally does not modify or overwrite existing PNGs produced earlier.
"""
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

METRICS = ['orig_skewness', 'orig_kurtosis', 'acorr_skewness_no_lag0', 'acorr_kurtosis_no_lag0']
FORCED_XLIMITS = {
    'orig_skewness': (-10.0, 10.0),
    'orig_kurtosis': (0.0, 50.0),
    # forced limits for autocorrelation-derived metrics
    'acorr_skewness_no_lag0': (-1.0, 80.0),
    'acorr_kurtosis_no_lag0': (-1.0, 4000.0),
}


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


def plot_metric(data, metric, fraction=1.0, seed=42):
    rng = np.random.RandomState(seed)
    rhythms = ['AF', 'NSR']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plt.suptitle(f'Normality checks: {metric} (fraction={fraction})')
    for i, ritmo in enumerate(rhythms):
        vals_full = data.loc[data['ritmo'] == ritmo, metric].dropna().values
        vals = sample_vals(vals_full, fraction, rng)
        ax_hist = axes[i, 0]
        ax_qq = axes[i, 1]
        if vals.size == 0:
            ax_hist.text(0.5, 0.5, 'No data', ha='center')
            ax_qq.text(0.5, 0.5, 'No data', ha='center')
            continue

        # Titles and labels
        ax_hist.set_title(f'{ritmo} (n={vals.size})')
        ax_hist.set_xlabel(metric.replace('_', ' ').title())
        ax_hist.set_ylabel('Density')

        # If there's a forced x-limit for this metric, use an explicit clipping approach
        if metric in FORCED_XLIMITS:
            rmin, rmax = map(float, FORCED_XLIMITS[metric])
            vals_clip = vals[(vals >= rmin) & (vals <= rmax)]
            if vals_clip.size == 0:
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
        else:
            # fallback: clip to 1-99 percentile central mass to avoid outlier domination
            try:
                p1, p99 = np.nanpercentile(vals, [1.0, 99.0])
                if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
                    raise Exception()
                span = p99 - p1
                rmin = p1 - 0.02 * span
                rmax = p99 + 0.02 * span
                vals_clip = vals[(vals >= rmin) & (vals <= rmax)]
                if vals_clip.size == 0:
                    sns.histplot(vals, bins=60, kde=True, ax=ax_hist, stat='density', color='tab:blue')
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
                    ax_hist.text(0.98, 0.95, 'Clipped to 1–99 pct', transform=ax_hist.transAxes, ha='right', va='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            except Exception:
                sns.histplot(vals, bins=60, kde=True, ax=ax_hist, stat='density', color='tab:blue')

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
    outfile = RESULTS_DIR / f'normality_{metric}_combined_frac{int(fraction*100)}.png'
    # avoid overwriting existing files: if exists, append a numeric suffix
    if outfile.exists():
        base = outfile.stem
        i = 1
        while True:
            candidate = RESULTS_DIR / f'{base}_{i}.png'
            if not candidate.exists():
                outfile = candidate
                break
            i += 1

    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', '-f', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data = load_and_normalize()
    outputs = []
    for metric in METRICS:
        if metric not in data.columns:
            print(f'Skipping {metric}: not in data')
            continue
        out = plot_metric(data, metric, fraction=args.fraction, seed=args.seed)
        outputs.append(str(out))
        print('Saved', out)
    print('\nAll combined plots saved.')


if __name__ == '__main__':
    main()
