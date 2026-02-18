#!/usr/bin/env python3
"""Quick probe: classify p09 records without saving signals.

Outputs a CSV with (record, tag, rmssd, cv, aux_excerpt) and prints summary stats.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import wfdb
import csv
import math


def is_NSR(aux_notes):
    if not aux_notes:
        return False
    inside_nsr = False
    for note in aux_notes:
        if note and str(note).startswith("(N"):
            inside_nsr = True
        elif note == ")" and inside_nsr:
            return True
    return False


def is_AF(aux_notes):
    if not aux_notes:
        return False
    inside_af = False
    for note in aux_notes:
        if note and (str(note).startswith("(AFIB") or str(note).startswith("(AF")):
            inside_af = True
        elif note == ")" and inside_af:
            return True
    return False


def compute_rr_metrics(samples, fs):
    if samples is None or len(samples) < 4:
        return (math.nan, math.nan)
    samples = np.asarray(samples)
    rr = np.diff(samples) / float(fs)
    if len(rr) < 3:
        return (math.nan, math.nan)
    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2)))
    mean_rr = float(np.mean(rr)) if np.mean(rr) > 0 else float('nan')
    cv = float(np.std(rr) / mean_rr) if mean_rr > 0 else float('nan')
    return (rmssd, cv)


def probe(in_root, out_csv, n_max=1000):
    in_root = Path(in_root)
    hea_files = list(in_root.rglob('*.hea'))
    rows = []
    counts = {'AF':0,'NSR':0,'UNK':0}
    for i, hea in enumerate(hea_files):
        if i >= n_max:
            break
        rec = str(hea.with_suffix(''))
        try:
            ann = wfdb.rdann(rec, 'atr')
        except Exception:
            rows.append((rec,'ERR',None,None,''))
            continue
        aux_notes = []
        try:
            if hasattr(ann, 'aux_note') and ann.aux_note is not None:
                if isinstance(ann.aux_note, (list,tuple)):
                    aux_notes = [str(a) for a in ann.aux_note if a]
                else:
                    aux_notes = [str(ann.aux_note)]
            elif hasattr(ann, 'comments') and ann.comments is not None:
                aux_notes = [str(c) for c in ann.comments if c] if isinstance(ann.comments,(list,tuple)) else [str(ann.comments)]
        except Exception:
            aux_notes = []
        tag = 'UNK'
        if is_AF(aux_notes):
            tag = 'AF'
        elif is_NSR(aux_notes):
            tag = 'NSR'
        # compute rr metrics
        samples = ann.sample if hasattr(ann, 'sample') else None
        fs = getattr(ann, 'fs', None)
        if fs is None:
            try:
                rec_r = wfdb.rdrecord(rec)
                fs = getattr(rec_r, 'fs', None) or 250.0
            except Exception:
                fs = 250.0
        rmssd, cv = compute_rr_metrics(samples, fs)
        # fallback heuristics if unk
        if tag == 'UNK' and not math.isnan(cv):
            if cv > 0.12 or rmssd > 0.08:
                tag = 'AF'
            elif cv < 0.05 and rmssd < 0.04:
                tag = 'NSR'
        counts[tag] = counts.get(tag,0) + 1
        excerpt = ' | '.join(aux_notes[:6]) if aux_notes else ''
        rows.append((rec, tag, rmssd, cv, excerpt))

    # save CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['record','tag','rmssd','cv','aux_excerpt'])
        for r in rows:
            w.writerow(r)

    # print summary
    total = len(rows)
    print(f'Probe processed {total} records')
    print('Counts:', counts)
    # compute basic stats
    rms = [r[2] for r in rows if r[2] is not None and not (isinstance(r[2], float) and math.isnan(r[2]))]
    cvs = [r[3] for r in rows if r[3] is not None and not (isinstance(r[3], float) and math.isnan(r[3]))]
    if rms:
        import numpy as _np
        print('rmssd: mean %.4f median %.4f' % (_np.mean(rms), _np.median(rms)))
    if cvs:
        import numpy as _np
        print('cv: mean %.4f median %.4f' % (_np.mean(cvs), _np.median(cvs)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input','-i', default='data/icentia/p09')
    p.add_argument('--out','-o', default='PULSOVITAL/npy_output_p09/classify_probe_2000.csv')
    p.add_argument('--n','-n', type=int, default=2000)
    args = p.parse_args()
    probe(args.input, args.out, n_max=args.n)


if __name__ == '__main__':
    main()
