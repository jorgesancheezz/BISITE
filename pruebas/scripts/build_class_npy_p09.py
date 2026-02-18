#!/usr/bin/env python3
"""Build consolidated class numpy files for p09 processed outputs.

Creates: PULSOVITAL/npy_output/AF_processed_1024x3000x1.npy
         PULSOVITAL/npy_output/NSR_processed_1024x3000x1.npy

Behavior:
- Prefer input root `PULSOVITAL/npy_output_p09_nsr_first` if present, else
  `PULSOVITAL/npy_output_p09`.
- For each class (AF, NSR) collects up to 1024 `_signal.npy` files,
  truncates or pads signals to length 3000, stacks into shape (N,3000,1),
  and saves dtype float32.
"""
import argparse
import os
from pathlib import Path
import numpy as np


def collect_signals(class_dir, max_samples=1024, target_len=3000):
    p = Path(class_dir)
    if not p.exists():
        return []
    files = sorted(list(p.rglob('*_signal.npy')))
    out = []
    for f in files[:max_samples]:
        try:
            a = np.load(f)
            a = np.asarray(a).ravel()
            if len(a) >= target_len:
                a = a[:target_len]
            else:
                pad = target_len - len(a)
                a = np.pad(a, (0, pad), mode='constant', constant_values=0.0)
            out.append(a.astype(np.float32))
        except Exception as e:
            print(f'Warning: failed load {f}: {e}')
    return out


def build_and_save(in_root, out_root, max_samples=1024, target_len=3000):
    in_root = Path(in_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for cls in ['AF', 'NSR']:
        class_dir = in_root / cls
        sigs = collect_signals(class_dir, max_samples=max_samples, target_len=target_len)
        if len(sigs) == 0:
            print(f'No signals found for class {cls} in {class_dir}')
            results[cls] = None
            continue
        arr = np.stack(sigs, axis=0)  # shape (N, target_len)
        arr = arr.reshape((arr.shape[0], target_len, 1))
        out_path = out_root / f"{cls}_processed_{max_samples}x{target_len}x1.npy"
        np.save(out_path, arr)
        print(f'Saved {cls} array with shape {arr.shape} -> {out_path}')
        results[cls] = str(out_path)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-root', default=None, help='Input processed p09 root')
    parser.add_argument('--out-root', default='PULSOVITAL/npy_output', help='Output folder for consolidated npy')
    parser.add_argument('--max', type=int, default=1024, help='Max samples per class')
    parser.add_argument('--len', type=int, default=3000, help='Target signal length (samples)')
    args = parser.parse_args()

    # auto-detect input root if not provided
    if args.in_root:
        in_root = Path(args.in_root)
    else:
        cand1 = Path('PULSOVITAL') / 'npy_output_p09_nsr_first'
        cand2 = Path('PULSOVITAL') / 'npy_output_p09'
        if cand1.exists():
            in_root = cand1
        else:
            in_root = cand2

    if not in_root.exists():
        print('No input root found for p09 processed outputs. Looked for:', str(in_root))
        return

    print('Using input root:', str(in_root))
    results = build_and_save(in_root, args.out_root, max_samples=args.max, target_len=args.len)
    print('Done. Outputs:')
    for k,v in results.items():
        print(' ', k, '->', v)


if __name__ == '__main__':
    main()
