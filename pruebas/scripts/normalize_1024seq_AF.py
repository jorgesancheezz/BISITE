#!/usr/bin/env python3
"""Normalize 1024seq_AF.npy in-place with a backup.

Produces a backup `1024seq_AF.npy.bak` and writes a per-sample z-normalized
version (float32) replacing the original file. NaNs are replaced with 0 before
normalization.
"""
import os
import shutil
import numpy as np


def main(p='1024seq_AF.npy'):
    if not os.path.exists(p):
        print('File not found:', p)
        return 1
    bak = p + '.bak'
    if not os.path.exists(bak):
        shutil.copy(p, bak)
        print('Backup created:', bak)
    else:
        print('Backup already exists:', bak)

    a = np.load(p)
    orig_shape = a.shape

    # collapse trailing channel dim if present
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]

    if a.ndim != 2:
        raise ValueError('Unexpected array shape: %s' % (orig_shape,))

    # replace NaNs
    nan_count = int(np.isnan(a).sum())
    if nan_count:
        print('Found NaNs:', nan_count, '-> replacing with 0')
    a = np.nan_to_num(a, nan=0.0)

    # per-sample (row) z-normalization
    means = a.mean(axis=1, keepdims=True)
    stds = a.std(axis=1, keepdims=True)
    eps = 1e-8
    stds_safe = stds.copy()
    stds_safe[stds_safe < eps] = 1.0
    a_norm = (a - means) / stds_safe

    # restore channel dim
    a_out = a_norm.reshape(a_norm.shape[0], a_norm.shape[1], 1).astype('float32')

    np.save(p, a_out)
    # also save a separate normalized copy for downstream workflows
    norm_copy = os.path.splitext(p)[0] + '_normalized.npy'
    np.save(norm_copy, a_out)
    print('Saved normalized file to', p)
    print('Also saved normalized copy to', norm_copy)
    print('Original shape:', orig_shape, '-> written shape:', a_out.shape, 'dtype:', a_out.dtype)
    # print some global stats
    print('Global min/max/mean/std:', float(a_out.min()), float(a_out.max()), float(a_out.mean()), float(a_out.std()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
