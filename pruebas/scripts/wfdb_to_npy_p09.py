#!/usr/bin/env python3
"""Convert icentia p09 WFDB records to per-record .npy/.json and a CSV summary.

Saves outputs under `PULSOVITAL/npy_output_p09/{AF,NSR,UNK}` and writes
`processed_records_p09.csv` in the same output root.

This script is robust to missing annotations and prints a final summary.
"""
import re
import os
import sys
import json
from pathlib import Path
import argparse
import traceback

import numpy as np
try:
    import wfdb
except Exception as e:
    print('wfdb import error:', e)
    raise


def is_NSR(aux_notes):
    """
    NSR según la regla usada en p10:
    Debe aparecer un texto que comience por '(N' y en algún punto aparezca ')'.
    """
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
    """
    AF según la regla usada en p10:
    Comienza '(AFIB' o '(AF' y termina con ')'.
    """
    if not aux_notes:
        return False

    inside_af = False
    for note in aux_notes:
        if note and (str(note).startswith("(AFIB") or str(note).startswith("(AF")):
            inside_af = True
        elif note == ")" and inside_af:
            return True

    return False


def safe_name(p: Path, base_dir: Path):
    # create a unique safe name relative to base_dir
    try:
        rel = p.relative_to(base_dir)
    except Exception:
        rel = p
    return str(rel).replace(os.sep, '__')


def process_record(hea_path: Path, in_root: Path, out_root: Path, nsr_first: bool=False):
    rec_base = hea_path.with_suffix('')
    rec_str = str(rec_base)
    entry = {'record': rec_str, 'tag': 'UNK', 'out_folder': None, 'signal_shape': None, 'error': None}
    try:
        rec = wfdb.rdrecord(rec_str)
        # p_signal preferred (float), fallback to d_signal
        sig = None
        if hasattr(rec, 'p_signal') and rec.p_signal is not None:
            sig = rec.p_signal
        elif hasattr(rec, 'd_signal') and rec.d_signal is not None:
            sig = rec.d_signal
        else:
            raise RuntimeError('No p_signal/d_signal found')
        sig = np.asarray(sig)
        # flatten to 1D if multi-channel with single channel
        if sig.ndim == 2 and sig.shape[1] == 1:
            sig = sig[:, 0]
        # read annotation if present and try multiple fallbacks to extract textual cues
        aux_notes = []
        ann_samples = np.array([], dtype=int)
        ann_symbols = np.array([], dtype=object)
        ann = None
        try:
            ann = wfdb.rdann(rec_str, extension='atr')
        except Exception:
            ann = None

        if ann is not None:
            try:
                if hasattr(ann, 'aux_note') and ann.aux_note is not None:
                    if isinstance(ann.aux_note, (list, tuple)):
                        aux_notes = [str(a) for a in ann.aux_note if a]
                    else:
                        aux_notes = [str(ann.aux_note)]
                elif hasattr(ann, 'comments') and ann.comments is not None:
                    aux_notes = [str(c) for c in ann.comments if c] if isinstance(ann.comments, (list,tuple)) else [str(ann.comments)]
            except Exception:
                aux_notes = []
            try:
                if hasattr(ann, 'sample'):
                    ann_samples = np.asarray(ann.sample)
                if hasattr(ann, 'symbol'):
                    ann_symbols = np.asarray(ann.symbol)
            except Exception:
                ann_samples = np.array([], dtype=int)
                ann_symbols = np.array([], dtype=object)

        # If aux_notes is empty, inspect annotation symbols for AF-like markers
        if (not aux_notes or len(aux_notes)==0) and len(ann_symbols) > 0:
            try:
                sym_join = ''.join([str(s) for s in ann_symbols if s is not None])
                if re.search(r'afib|atrial fibrill|af', sym_join, re.IGNORECASE):
                    aux_notes = ['AF_from_symbols']
            except Exception:
                pass

        # If still empty, attempt to read textual annotation files in the same folder
        if not aux_notes:
            for ext in ('.atr', '.atr.txt', '.ann', '.cls', '.txt'):
                p = hea_path.with_suffix(ext)
                if p.exists():
                    try:
                        txt = p.read_text(errors='ignore')
                        if txt and len(txt) > 0:
                            # split into lines as aux notes
                            aux_notes = [l.strip() for l in txt.splitlines() if l.strip()]
                            break
                    except Exception:
                        continue

        # If we were able to extract textual cues, use them first.
        # Optionally allow checking NSR before AF when `nsr_first` is True.
        tag = 'UNK'
        if nsr_first:
            if is_NSR(aux_notes):
                tag = 'NSR'
            elif is_AF(aux_notes):
                tag = 'AF'
        else:
            if is_AF(aux_notes):
                tag = 'AF'
            elif is_NSR(aux_notes):
                tag = 'NSR'

        # If textual cues are absent, try to classify from beat annotations (.atr):
        # compute RR interval irregularity (RMSSD and coefficient of variation).
        try:
            if ann is not None and hasattr(ann, 'sample') and len(ann.sample) > 3:
                fs = getattr(rec, 'fs', None) or 250.0
                samples = np.asarray(ann.sample)
                rr = np.diff(samples) / float(fs)
                if len(rr) >= 3:
                    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2)))
                    mean_rr = float(np.mean(rr)) if np.mean(rr) > 0 else 0.0
                    cv = float(np.std(rr) / mean_rr) if mean_rr > 0 else 0.0
                    # heuristic thresholds (tunable):
                    # AF tends to have higher RR irregularity (rmssd and cv)
                    if cv > 0.12 or rmssd > 0.08:
                        tag = 'AF'
                    elif cv < 0.05 and rmssd < 0.04:
                        tag = 'NSR'
        except Exception:
            pass

        rel_safe = safe_name(hea_path.parent, in_root)
        fname_safe = safe_name(hea_path.with_suffix(''), in_root)
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save signal (float32)
        sig_path = out_dir / f'{fname_safe}_signal.npy'
        np.save(str(sig_path), sig.astype(np.float32), allow_pickle=False)

        # Save annotation arrays
        ann_samp_path = out_dir / f'{fname_safe}_annotation_samples.npy'
        ann_sym_path = out_dir / f'{fname_safe}_annotation_symbols.npy'
        np.save(str(ann_samp_path), ann_samples, allow_pickle=False)
        np.save(str(ann_sym_path), ann_symbols.astype(object), allow_pickle=True)

        # Save aux notes + metadata
        meta = {
            'record_path': rec_str,
            'sample_rate': getattr(rec, 'fs', None),
            'channels': sig.shape[1] if (sig.ndim==2) else 1,
            'signal_shape': sig.shape,
            'aux_notes_excerpt': (" | ").join(aux_notes[:10]) if aux_notes else ''
        }
        meta_path = out_dir / f'{fname_safe}_metadata.json'
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

        entry.update({'tag': tag, 'out_folder': str(out_dir), 'signal_shape': str(sig.shape)})
        return entry
    except Exception as e:
        entry['error'] = repr(e)
        entry['traceback'] = traceback.format_exc()
        return entry


def find_hea_files(in_root: Path):
    # find .hea files under directory
    for p in sorted(in_root.rglob('*.hea')):
        yield p


def main():
    p = argparse.ArgumentParser(description='Convert p09 WFDB -> npy')
    p.add_argument('--input', '-i', default='data/icentia/p09', help='p09 root with per-record folders (contains .hea/.dat/.atr)')
    p.add_argument('--out', '-o', default='PULSOVITAL/npy_output_p09', help='Output root for processed npy files')
    p.add_argument('--max', type=int, default=0, help='Max records to process (0=all)')
    p.add_argument('--skip-unk', action='store_true', help='Skip saving UNK records')
    p.add_argument('--nsr-first', action='store_true', help='When present, check NSR rule before AF (marks NSR first)')
    p.add_argument('--csv-name', default=None, help='Optional CSV filename (inside --out). Default: processed_records_p09.csv')
    args = p.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    hea_files = list(find_hea_files(in_root))
    n_total = len(hea_files)
    if args.max and args.max > 0:
        hea_files = hea_files[:args.max]

    processed = []
    errors = []
    counts = {'AF':0, 'NSR':0, 'UNK':0}

    print(f'Found {n_total} .hea files under {in_root}. Processing {len(hea_files)} records...')

    for i, hea in enumerate(hea_files, 1):
        sys.stdout.write(f'[{i}/{len(hea_files)}] {hea} ... ')
        sys.stdout.flush()
        entry = process_record(hea, in_root, out_root, nsr_first=args.nsr_first)
        processed.append(entry)
        if entry.get('error'):
            errors.append(entry)
            print('ERROR')
        else:
            counts[entry.get('tag','UNK')] = counts.get(entry.get('tag','UNK'),0) + 1
            print(entry.get('tag'))

    # write CSV summary
    # write CSV summary (allow custom name when running nsr-first)
    try:
        import pandas as _pd
        df = _pd.DataFrame(processed)
        if args.csv_name:
            csv_path = out_root / args.csv_name
        else:
            default_name = 'processed_records_p09_nsr_first.csv' if args.nsr_first else 'processed_records_p09.csv'
            csv_path = out_root / default_name
        df.to_csv(csv_path, index=False)
    except Exception:
        # fallback to json
        with open(out_root / 'processed_records_p09.json', 'w', encoding='utf-8') as fh:
            json.dump(processed, fh, indent=2, ensure_ascii=False)
        csv_path = out_root / 'processed_records_p09.json'

    print('\nProcessing complete.')
    print('Counts:', counts)
    print('Total processed:', len(processed), 'Errors:', len(errors))
    print('Summary saved to:', str(csv_path))
    os.system('Get-Content PULSOVITAL\\npy_output_p09_nsr_first\\process_p09.log -Tail 200 -Wait')


if __name__ == '__main__':
    main()
