#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
from numpy.random import default_rng

rng = default_rng(42)

in_root = Path('PULSOVITAL/npy_output_p09_consolidated')
files = {
    'AF': in_root / 'AF_processed_1024x3000x1.npy',
    'NSR': in_root / 'NSR_processed_1024x3000x1.npy'
}

outs = {
    'AF': in_root / '1024seq_AF.npy',
    'NSR': in_root / '1024seq_NSR.npy'
}

# Backup existing root files if present
for name in ['1024seq_AF.npy','1024seq_NSR.npy']:
    p = Path(name)
    if p.exists():
        b = p.with_suffix('.orig.npy')
        try:
            if b.exists():
                b.unlink()
            p.replace(b)
            print(f'Backed up {p} -> {b}')
        except Exception as e:
            print('  backup failed:', e)

for k,v in files.items():
    if not v.exists():
        print('Input missing for', k, ':', v)
        continue
    arr = np.load(v)
    n = arr.shape[0]
    print(k, 'loaded shape', arr.shape)
    if n >= 1024:
        out = arr[:1024]
    else:
        idx = rng.choice(n, size=1024, replace=True)
        out = arr[idx]
    out = out.astype(np.float32)
    # save to consolidated folder
    outp = outs[k]
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, out)
    print('Saved upsampled', k, '->', outp, 'shape', out.shape)
    # also save to repo root filename
    rootp = Path(f'1024seq_{k}.npy')
    np.save(rootp, out)
    print('Also saved copy to', rootp)

print('Done')
