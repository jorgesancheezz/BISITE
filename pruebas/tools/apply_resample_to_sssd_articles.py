"""Aplica resampling a `sssd_article_AF.npy` y `sssd_article_NSR.npy`.

Por defecto usa `original_freq=100` y `target_freq=300` (enteros).
Hace backup de los .npy originales antes de sobrescribirlos y escribe
`resampling-config.yaml` con los parámetros.
"""
import argparse
import os
import shutil
import time
from pathlib import Path

import yaml
import numpy as np
from scipy.signal import resample_poly


def backup(path: Path):
    if path.exists():
        ts = int(time.time())
        bak = path.with_suffix(path.suffix + f'.bak.{ts}')
        shutil.move(str(path), str(bak))
        print(f'Backed up {path} -> {bak}')


def resample_array(arr: np.ndarray, up: int, down: int) -> np.ndarray:
    # apply resample_poly along axis=1 (time axis)
    if arr.ndim == 3:
        xr = resample_poly(arr, up, down, axis=1)
    elif arr.ndim == 2:
        xr = resample_poly(arr, up, down, axis=1)
        # keep (N, T) shape
    else:
        # 1D signal
        xr = resample_poly(arr, up, down)
    return xr


def main(args):
    files = [Path(p) for p in args.files]

    cfg = {
        'files': [str(p) for p in files],
        'original_freq': int(args.original_freq),
        'target_freq': int(args.target_freq),
    }
    # write config
    with open(os.path.join(args.output_dir, 'resampling-config.yaml'), 'w') as fd:
        yaml.dump(cfg, fd)

    up = int(args.target_freq)
    down = int(args.original_freq)

    for p in files:
        if not p.exists():
            print(f'File not found: {p} — skipping')
            continue

        print(f'Processing {p}...')
        arr = np.load(p)
        xr = resample_array(arr, up, down)

        # convert to channel-last (N,T,C) if necessary
        if xr.ndim == 2:
            # (N, T) -> (N, T, 1)
            xr = xr.reshape((xr.shape[0], xr.shape[1], 1))
        elif xr.ndim == 1:
            xr = xr.reshape((1, xr.shape[0], 1))

        # cast to float16 to save space (match earlier pipeline)
        xr_out = xr.astype(np.float16)

        # backup and overwrite
        backup(p)
        np.save(p, xr_out)
        print(f'Wrote {p} shape={xr_out.shape} dtype={xr_out.dtype}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', default=['sssd_article_AF.npy', 'sssd_article_NSR.npy'])
    parser.add_argument('--original-freq', type=int, default=100)
    parser.add_argument('--target-freq', type=int, default=300)
    parser.add_argument('--output-dir', type=str, default='.')
    args = parser.parse_args()
    main(args)
