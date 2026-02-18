#!/usr/bin/env python3
"""Postprocess generated sample .npy files from SSSD-ECG inference to shape (N=1024, T=3000, C=1).
Usage:
  python scripts/postprocess_generated_samples.py --ckpt-dir sssd_label_cond/ch256_T200_betaT0.02 --out generated_1024x3000x1.npy --lead 0
"""
import argparse
import os
import glob
import numpy as np
import math


def load_all_samples(ckpt_dir):
    files = sorted(glob.glob(os.path.join(ckpt_dir, '*_samples.npy')))
    if not files:
        raise FileNotFoundError(f'No *_samples.npy found in {ckpt_dir}')
    parts = [np.load(f) for f in files]
    return np.concatenate(parts, axis=0)


def resample_time_axis(x, target_T):
    # x shape: (N, T)
    N, T = x.shape
    if T == target_T:
        return x
    if target_T % T == 0:
        reps = target_T // T
        return np.tile(x, (1, reps))
    xp = np.arange(T)
    x_new = np.linspace(0, T - 1, target_T)
    y_res = np.zeros((N, target_T), dtype=x.dtype)
    for i in range(N):
        y_res[i] = np.interp(x_new, xp, x[i])
    return y_res


def ensure_N(x, target_N):
    # x shape (M, T)
    M = x.shape[0]
    if M == target_N:
        return x
    if M < target_N:
        times = int(math.ceil(target_N / M))
        return np.tile(x, (times, 1))[:target_N]
    return x[:target_N]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', required=True, help='Directory where inference saved *_samples.npy')
    p.add_argument('--out', default='generated_1024x3000x1.npy')
    p.add_argument('--lead', type=int, default=0, help='Lead index to extract from generated 12-lead (0-based)')
    p.add_argument('--N', type=int, default=1024)
    p.add_argument('--T', type=int, default=3000)
    args = p.parse_args()

    all_samples = load_all_samples(args.ckpt_dir)
    # expected shape: (M, channels, 1000)
    if all_samples.ndim != 3:
        raise ValueError('Unexpected samples array shape: %s' % (all_samples.shape,))
    M, C, orig_T = all_samples.shape
    if args.lead >= C:
        raise ValueError('lead index out of range: %d >= %d' % (args.lead, C))

    # select lead
    lead_data = all_samples[:, args.lead, :]

    # resample time axis to T
    lead_res = resample_time_axis(lead_data, args.T)

    # ensure N samples
    lead_res = ensure_N(lead_res, args.N)

    # final shape (N, T, 1)
    out_arr = lead_res.reshape((args.N, args.T, 1)).astype(np.float32)
    np.save(args.out, out_arr)
    print('Saved', args.out, 'shape=', out_arr.shape)

if __name__ == '__main__':
    main()
