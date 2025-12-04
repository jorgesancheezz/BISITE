#!/usr/bin/env python3
"""Diagnose p09 label issues and produce suggested relabels.

Writes two CSVs into the repo root:
- p09_label_diagnosis.csv  : full diagnostic table
- p09_suggested_relabel.csv: only rows where suggested_label != current_label or discard

Conservative rules (as requested):
- If annotation contains 'AF' or 'AFIB' (case-ins), treat as AF.
- If annotation is exactly 'A' (or contains standalone ' A '), do NOT treat as AF.
- Otherwise use RR coefficient-of-variation heuristic (default threshold 0.12) to suggest AF.
"""
import os
import sys
import glob
import argparse
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def compute_rr_cv(sig, fs=300.0):
    sig = np.asarray(sig).ravel()
    if len(sig) < 3:
        return float('nan')
    # simple peak detection on the signal magnitude
    try:
        peaks, _ = find_peaks(sig, distance=int(0.3*fs))
    except Exception:
        peaks = np.array([])
    if len(peaks) < 2:
        return float('nan')
    rr = np.diff(peaks) / float(fs)
    if np.mean(rr) == 0:
        return float('nan')
    return float(np.std(rr) / np.mean(rr))


def nan_fraction(sig):
    a = np.asarray(sig)
    return float(np.isnan(a).sum()) / max(1, a.size)


def infer_records_from_dirs(base_dir='PULSOVITAL'):
    # look for per-record npy files under common p09 output dirs
    patterns = [
        os.path.join(base_dir, 'npy_output_p09', '**', '*.npy'),
        os.path.join(base_dir, 'npy_output_p09_consolidated', '*.npy'),
        os.path.join(base_dir, '**', 'p09', '**', '*.npy'),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))
    recs = []
    for p in files:
        # infer current_label from parent folder names
        parts = p.replace('\\','/').split('/')
        label = 'UNKNOWN'
        for part in parts[::-1]:
            up = part.lower()
            if up in ('af','af_real','af_synth','af_processed'):
                label = 'AF'
                break
            if up in ('nsr','nsr_real','nsr_synth','nsr_processed'):
                label = 'NSR'
                break
        recs.append({'record': os.path.splitext(os.path.basename(p))[0], 'current_label': label, 'path': p, 'annotation': ''})
    return recs


def load_processed_csv(path):
    try:
        df = pd.read_csv(path)
        # expected columns: record, label, aux_note, path (best-effort)
        records = []
        for _, r in df.iterrows():
            rec = dict(record=str(r.get('record', '')), current_label=str(r.get('label', '')).upper() if not pd.isna(r.get('label', '')) else 'UNKNOWN', path=str(r.get('path', '')) if not pd.isna(r.get('path', '')) else '', annotation=str(r.get('aux_note', r.get('annotation', '')) if not pd.isna(r.get('aux_note', '')) else ''))
            records.append(rec)
        return records
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rr_cv_thresh', type=float, default=0.12)
    p.add_argument('--fs', type=float, default=300.0)
    p.add_argument('--out_prefix', type=str, default='p09')
    args = p.parse_args()

    processed_csv = os.path.join('PULSOVITAL', 'npy_output_p09', 'processed_records_p09.csv')
    records = []
    if os.path.exists(processed_csv):
        records = load_processed_csv(processed_csv)
        print(f'Loaded {len(records)} records from {processed_csv}')
    else:
        print('processed_records_p09.csv not found; scanning for .npy files under PULSOVITAL to infer records...')
        records = infer_records_from_dirs('PULSOVITAL')
        print(f'Inferred {len(records)} candidate records from filesystem')

    rows = []
    for rec in records:
        path = rec.get('path','')
        sig = None
        if path and os.path.exists(path):
            try:
                sig = np.load(path)
            except Exception:
                try:
                    # maybe the saved file is an array inside consolidated; skip if cannot load
                    sig = None
                except Exception:
                    sig = None
        else:
            # try to locate using record name
            candidate = None
            base = rec.get('record','')
            for c in glob.glob(os.path.join('PULSOVITAL','**', base + '.npy'), recursive=True):
                candidate = c; break
            if candidate:
                try:
                    sig = np.load(candidate); path = candidate
                except Exception:
                    sig = None

        rr_cv = float('nan'); nf = float('nan')
        if sig is not None:
            rr_cv = compute_rr_cv(sig, fs=args.fs)
            nf = nan_fraction(sig)

        ann = (rec.get('annotation') or '').upper()
        current = (rec.get('current_label') or 'UNKNOWN')

        # rule: AF if annotation contains 'AF' or 'AFIB'
        if 'AFIB' in ann or 'AF' in ann:
            suggested = 'AF'
            reason = 'annotation_AF/AFIB'
        else:
            # if annotation contains standalone ' A ' or equals 'A', do NOT mark AF
            tokens = [t.strip() for t in ann.replace(',', ' ').split() if t.strip()]
            if any(t == 'A' for t in tokens) and not any('AF' in t for t in tokens):
                suggested = 'NSR'
                reason = 'annotation_A_excluded'
            else:
                # heuristic by rr_cv
                if np.isnan(rr_cv):
                    suggested = current  # unknown -> leave as-is
                    reason = 'no_peaks_or_short_signal'
                else:
                    if rr_cv > args.rr_cv_thresh:
                        suggested = 'AF'
                        reason = f'rr_cv>{args.rr_cv_thresh}'
                    else:
                        suggested = 'NSR'
                        reason = f'rr_cv<={args.rr_cv_thresh}'

        # discard if too many NaNs or tiny signal
        if not np.isnan(nf) and nf > 0.10:
            suggested = 'DISCARD'
            reason = 'high_nan_fraction'

        rows.append({'record': rec.get('record',''), 'current_label': current, 'suggested_label': suggested, 'reason': reason, 'rr_cv': rr_cv, 'nan_fraction': nf, 'annotation': rec.get('annotation',''), 'path': path})

    df = pd.DataFrame(rows)
    full_out = args.out_prefix + '_label_diagnosis.csv'
    sugg_out = args.out_prefix + '_suggested_relabel.csv'
    df.to_csv(full_out, index=False)
    df[df['suggested_label'] != df['current_label']].to_csv(sugg_out, index=False)
    print(f'Wrote {full_out} ({len(df)} rows) and {sugg_out} ({(df["suggested_label"] != df["current_label"]).sum()} suggested changes)')


if __name__ == '__main__':
    main()
