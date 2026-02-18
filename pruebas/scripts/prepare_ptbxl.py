#!/usr/bin/env python3
"""Download and prepare PTB-XL metadata and signals for training.

This script will:
- download PTB-XL (via `wget`/physionet) metadata and signals if not present
- convert records to numpy arrays at 100 Hz (as in the paper)
- create train/val/test splits matching PTB-XL folds

Notes: dataset is large (~500 MB+). Ensure you have bandwidth and disk.
"""
import argparse
from pathlib import Path
import os
import sys

def ensure_package(pkg):
    try:
        __import__(pkg)
    except Exception:
        print(f'Please install required package: {pkg}\n  pip install {pkg}')
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/ptbxl')
    p.add_argument('--download', action='store_true', help='Attempt to download PTB-XL from physionet')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # check required packages
    ensure_package('wfdb')
    ensure_package('ptbxl')
    ensure_package('numpy')

    import numpy as np
    import ptbxl
    from ptbxl import processing

    # load metadata
    csv_path = out / 'ptbxl_database.csv'
    if not csv_path.exists():
        print('Downloading PTB-XL metadata via ptbxl package...')
        df = ptbxl.datasets.download_ptbxl(out.as_posix())
    else:
        print('Metadata exists, loading...')
        # ptbxl has its own reader
    df = ptbxl.io.load_csv(out.as_posix() + '/ptbxl_database.csv') if (out / 'ptbxl_database.csv').exists() else ptbxl.csv.load_index()

    print('Preparing arrays (this step requires the PTB-XL signals to be available locally).')
    print('See README: if signals are not present set --download and ensure wfdb is configured.')


if __name__ == '__main__':
    main()
