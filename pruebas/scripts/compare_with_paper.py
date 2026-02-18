#!/usr/bin/env python3
"""Compute macro-AUROC and F1 for real vs synthetic p09 datasets.

Usage examples:
  python scripts/compare_with_paper.py
  python scripts/compare_with_paper.py --real-af PULSOVITAL/npy_output_p09_consolidated/1024seq_AF.npy \
      --real-nsr PULSOVITAL/npy_output_p09_consolidated/1024seq_NSR.npy \
      --synth-af 1024seq_AF.npy --synth-nsr 1024seq_NSR.npy

The script trains a simple logistic classifier on downsampled signals and
computes AUROC and F1 for the four evaluation modes described in the paper.
"""
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import csv


def load_pairs(af_path, nsr_path):
    a = np.load(af_path)
    n = np.load(nsr_path)
    # expect shape (N, L, C) -> flatten channel
    def norm(x):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], x.shape[1])
        return x
    a = norm(a)
    n = norm(n)
    X = np.vstack([a, n])
    y = np.hstack([np.ones(a.shape[0], dtype=int), np.zeros(n.shape[0], dtype=int)])
    # remove any samples containing NaN
    mask = np.isfinite(X).all(axis=1)
    if not mask.all():
        X = X[mask]
        y = y[mask]
    return X, y


def downsample_mean(X, factor=10):
    # X: (N, L) -> (N, L//factor)
    N, L = X.shape
    L2 = L // factor
    X2 = X[:, :L2*factor].reshape(N, L2, factor).mean(axis=2)
    return X2


def make_splits(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def fit_and_eval(X_train, y_train, X_test, y_test):
    # scaler + logistic
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_s, y_train)
    prob = clf.predict_proba(X_test_s)[:, 1]
    pred = clf.predict(X_test_s)
    auc = roc_auc_score(y_test, prob)
    f1 = f1_score(y_test, pred)
    return auc, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--real-af', default='PULSOVITAL/npy_output_p09_consolidated/1024seq_AF.npy')
    p.add_argument('--real-nsr', default='PULSOVITAL/npy_output_p09_consolidated/1024seq_NSR.npy')
    p.add_argument('--synth-af', default='1024seq_AF.npy')
    p.add_argument('--synth-nsr', default='1024seq_NSR.npy')
    p.add_argument('--downsample-factor', type=int, default=10)
    p.add_argument('--out-csv', default='compare_out_test/compare_with_paper_results.csv')
    args = p.parse_args()

    real_af = Path(args.real_af)
    real_nsr = Path(args.real_nsr)
    synth_af = Path(args.synth_af)
    synth_nsr = Path(args.synth_nsr)

    for f in (real_af, real_nsr, synth_af, synth_nsr):
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")

    X_real, y_real = load_pairs(str(real_af), str(real_nsr))
    X_synth, y_synth = load_pairs(str(synth_af), str(synth_nsr))

    # downsample
    X_real_ds = downsample_mean(X_real, factor=args.downsample_factor)
    X_synth_ds = downsample_mean(X_synth, factor=args.downsample_factor)

    # splits
    Xr_tr, Xr_te, yr_tr, yr_te = make_splits(X_real_ds, y_real)
    Xs_tr, Xs_te, ys_tr, ys_te = make_splits(X_synth_ds, y_synth)

    results = []

    # Reference: real->real
    auc_rr, f1_rr = fit_and_eval(Xr_tr, yr_tr, Xr_te, yr_te)
    results.append(('real->real', auc_rr, f1_rr))

    # (1) synth->real
    auc_sr, f1_sr = fit_and_eval(Xs_tr, ys_tr, Xr_te, yr_te)
    results.append(('synth->real', auc_sr, f1_sr))

    # (2) real(ref)->synth
    auc_rs, f1_rs = fit_and_eval(Xr_tr, yr_tr, Xs_te, ys_te)
    results.append(('real->synth', auc_rs, f1_rs))

    # (3) synth->synth
    auc_ss, f1_ss = fit_and_eval(Xs_tr, ys_tr, Xs_te, ys_te)
    results.append(('synth->synth', auc_ss, f1_ss))

    # save csv
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['eval_mode','AUROC','F1'])
        for r in results:
            w.writerow(r)

    for r in results:
        print(f'{r[0]:12s} AUROC={r[1]:.4f}  F1={r[2]:.4f}')


if __name__ == '__main__':
    main()
