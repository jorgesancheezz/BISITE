#!/usr/bin/env python3
"""Download PTB-XL metadata and signals (using wfdb remote access).

By default this script downloads the PTB-XL metadata CSV and the records
belonging to folds 9 and 10 (validation and test as used in many papers).
Set --folds to download other folds (comma-separated or 'all').

Signals are saved under `data/ptbxl/signals/{ecg_id}.npy` and the metadata
CSV is saved to `data/ptbxl/ptbxl_database.csv`.
"""
import argparse
from pathlib import Path
import requests
import csv
import wfdb
import time
import numpy as np


def download_csv(out_path):
    url = 'https://physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv?download'
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f'Failed to download CSV {url} status={r.status_code}')
    out_path.write_bytes(r.content)


def load_csv(path):
    import pandas as pd
    return pd.read_csv(path)


def save_record_signal(ecg_id, out_dir):
    # ecg_id is the record name as in the CSV (integer or string like '1')
    rec_name = str(ecg_id)
    try:
        rec = wfdb.rdrecord(rec_name, pn_dir='ptb-xl/1.0.1')
    except Exception as e:
        print('Failed to read record', rec_name, e)
        return False
    # p_signal is NxM (N samples, M leads)
    sig = rec.p_signal.astype(np.float32)
    # save as (M, N) for convenience
    sig = sig.T
    outp = out_dir / f'{rec_name}.npy'
    np.save(outp, sig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/ptbxl')
    p.add_argument('--folds', default='9,10', help="comma list or 'all'")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csvp = out / 'ptbxl_database.csv'
    if not csvp.exists():
        print('Downloading PTB-XL metadata CSV...')
        download_csv(csvp)
    else:
        print('Metadata CSV already present at', csvp)

    print('Loading CSV...')
    df = load_csv(csvp)
    # ensure strat_fold exists
    if 'strat_fold' not in df.columns:
        print('CSV missing strat_fold; ensure correct PTB-XL CSV version')

    if args.folds.strip().lower() == 'all':
        folds = sorted(df['strat_fold'].unique())
    else:
        folds = [int(x) for x in args.folds.split(',')]

    sigdir = out / 'signals'
    sigdir.mkdir(exist_ok=True)

    total = 0
    for f in folds:
        sub = df[df['strat_fold'] == f]
        print(f'Processing fold {f} with {len(sub)} records')
        for i, row in sub.iterrows():
            rec = row['ecg_id']
            outp = sigdir / f'{rec}.npy'
            if outp.exists():
                total += 1
                continue
            ok = save_record_signal(rec, sigdir)
            if not ok:
                print('Skipping', rec)
            else:
                total += 1
            time.sleep(0.05)

    print('Downloaded signals count=', total)


if __name__ == '__main__':
    main()
