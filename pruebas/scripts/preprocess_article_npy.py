#!/usr/bin/env python3
"""Preprocesado simple para los .npy del artículo.

Carga archivos .npy (forma esperada (N, L, 1) o (N, L)), remuestrea de
`original_freq` a `target_freq` usando `scipy.signal.resample_poly` con
una aproximación racional, convierte a `float16` y guarda los arrays
procesados en `out_dir` conservando el nombre con sufijo `_proc.npy`.

Ejemplo:
  python scripts/preprocess_article_npy.py --inputs sssd_article_AF.npy sssd_article_NSR.npy \
      --original-freq 300 --target-freq 100 --out-dir PULSOVITAL/Metricas

"""
import argparse
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly


def process_file(p_in: Path, original_freq: float, target_freq: float, out_dir: Path, copy_shape3: bool = True):
    arr = np.load(p_in)
    if arr.ndim == 3 and arr.shape[2] == 1:
        squeezed = arr.reshape(arr.shape[0], arr.shape[1])
        has_channel = True
    elif arr.ndim == 2:
        squeezed = arr
        has_channel = False
    else:
        # try to handle other shapes by flattening trailing dims
        squeezed = arr.reshape(arr.shape[0], -1)
        has_channel = squeezed.shape[1] != arr.shape[1]

    # compute integer up/down from rational approx
    frac = Fraction(str(target_freq)) / Fraction(str(original_freq))
    frac = frac.limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    # perform resampling along axis=1
    try:
        res = resample_poly(squeezed, up, down, axis=1)
    except Exception as e:
        # fallback to target length estimation + linear interpolation
        L = squeezed.shape[1]
        newL = int(round(L * (float(target_freq) / float(original_freq))))
        x = np.linspace(0, 1, L)
        xp = np.linspace(0, 1, newL)
        res = np.array([np.interp(xp, x, row) for row in squeezed])

    # restore 3rd dim if necessary
    if has_channel:
        res = res.reshape(res.shape[0], res.shape[1], 1)

    # cast to float16 to match other pipeline behavior
    res = res.astype(np.float16)

    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / (p_in.stem + '_proc.npy')
    np.save(outp, res)
    return outp, res.shape, res.dtype


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True, help='Lista de archivos .npy a procesar')
    p.add_argument('--original-freq', type=float, required=True)
    p.add_argument('--target-freq', type=float, required=True)
    p.add_argument('--out-dir', type=str, default='processed_article', help='Carpeta de salida')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    results = []
    for fp in args.inputs:
        p_in = Path(fp)
        if not p_in.exists():
            print(f'Missing input: {p_in} -- skipping')
            continue
        outp, shape, dtype = process_file(p_in, args.original_freq, args.target_freq, out_dir)
        print(f'Processed {p_in} -> {outp}  shape={shape} dtype={dtype}')
        results.append(str(outp))

    if len(results) == 0:
        print('No files processed.')


if __name__ == '__main__':
    main()
