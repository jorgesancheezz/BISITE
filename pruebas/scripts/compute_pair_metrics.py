#!/usr/bin/env python3
"""Compute table of metrics for AF and NSR pairs between 1024seq and article processed files.

Uses functions from tools/compare_with_article_demos.py logic (reimplemented minimal helpers)
and writes CSV to `notebooks/outputs/pair_metrics_table.csv` and prints a simple ASCII table.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.signal import welch
from scipy.stats import ks_2samp

# import local helpers from tools/compare_with_article_demos if available
from tools.compare_with_article_demos import prepare, resample_to, compute_all_metrics


PAIRS = [
    (Path('PULSOVITAL/Metricas/1024seq_AF.npy'), Path('PULSOVITAL/Metricas/sssd_article_AF_proc_3000.npy'), 'AF'),
    (Path('PULSOVITAL/Metricas/1024seq_NSR.npy'), Path('PULSOVITAL/Metricas/sssd_article_NSR_proc_3000.npy'), 'NSR'),
]


def main():
    out_dir = Path('notebooks/outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for a_path, b_path, cls in PAIRS:
        print('Processing', a_path, 'vs', b_path)
        if not a_path.exists():
            print('Missing', a_path); continue
        if not b_path.exists():
            print('Missing', b_path); continue
        A = prepare(np.load(a_path))
        B = prepare(np.load(b_path))
        # align lengths
        target_len = int(min(A.shape[1], B.shape[1]))
        A_rs = resample_to(A, target_len)
        B_rs = resample_to(B, target_len)
        metrics = compute_all_metrics(A_rs, B_rs, name=f'{a_path.name}_vs_{b_path.name}')
        # select desired columns
        row = {'CLASS': cls,
               'DS': metrics.get('DS', float('nan')),
               'PS': metrics.get('PS', float('nan')),
               'MDD': metrics.get('MDD', float('nan')),
               'ACD': metrics.get('ACD', float('nan')),
               'SD': metrics.get('SD', float('nan')),
               'KD': metrics.get('KD', float('nan')),
               'CFID': metrics.get('CFID', float('nan'))}
        rows.append(row)

    df = pd.DataFrame(rows)
    csvp = out_dir / 'pair_metrics_table.csv'
    df.to_csv(csvp, index=False)
    print('Saved CSV:', csvp)
    print(df.to_string(index=False, float_format='{:0.4f}'.format))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
