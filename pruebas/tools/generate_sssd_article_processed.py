"""Genera `sssd_article_AF.npy` y `sssd_article_NSR.npy` procesados

Creación de 1024 muestras por clase, longitud objetivo 3000 y 1 canal.
Se crea variación realista sencilla (escala, ruido, desplazamiento), AF recibe
mayor irregularidad. Si existen los archivos de salida, se guardan backups.
"""
import argparse
import os
import shutil
from pathlib import Path
import time

import numpy as np
from scipy.signal import butter, filtfilt


def _backup(path: Path):
    if path.exists():
        ts = int(time.time())
        bak = path.with_suffix(path.suffix + f'.bak.{ts}')
        shutil.move(str(path), str(bak))
        print(f'Backed up {path} -> {bak}')


def _resample_to_length(sig: np.ndarray, t_target: int) -> np.ndarray:
    # Linear interpolation resample (robusto y sin dependencias extra)
    t_orig = sig.shape[0]
    xp = np.arange(t_orig)
    x_new = np.linspace(0, t_orig - 1, t_target)
    return np.interp(x_new, xp, sig).astype(np.float32)


def _bandpass(sig: np.ndarray, fs: float = 300.0, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    # 4th-order Butterworth bandpass, applied with filtfilt for zero-phase
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    if low_n <= 0 or high_n >= 1 or low_n >= high_n:
        return sig
    b, a = butter(4, [low_n, high_n], btype='band')
    try:
        out = filtfilt(b, a, sig)
    except Exception:
        out = sig
    return out.astype(np.float32)


def _make_variants(base: np.ndarray, n: int, noise_scale: float, jitter_shift: int, seed: int = None, af: bool = False):
    rng = np.random.RandomState(seed)
    out = np.empty((n, base.shape[0]), dtype=np.float32)
    base_std = base.std() if base.std() > 0 else 1.0

    for i in range(n):
        x = base.copy()
        # amplitude variation
        amp = rng.uniform(0.9, 1.1)
        x = x * amp
        # random circular shift
        shift = rng.randint(-jitter_shift, jitter_shift + 1)
        if shift != 0:
            x = np.roll(x, shift)
        # additive gaussian noise
        sigma = noise_scale * base_std
        x = x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)

        if af:
            # AF: add occasional bursts and higher noise
            bursts = rng.randint(3, 12)
            for _ in range(bursts):
                start = rng.randint(0, x.shape[0])
                length = rng.randint(10, 200)
                end = min(x.shape[0], start + length)
                x[start:end] += rng.normal(0.0, 2.0 * sigma, size=(end - start,))
            # slight baseline wandering
            lw = rng.uniform(-0.01, 0.01, size=x.shape)
            x = x + lw * np.max(np.abs(base))

        # final clipping to plausible ECG amplitude range (relative)
        p = np.percentile(np.abs(base), 99)
        clip = max(5.0 * p, 1e-3)
        x = np.clip(x, -clip, clip)

        out[i] = x.astype(np.float32)

    return out


def main(args):
    src = Path(args.source)
    if not src.exists():
        raise FileNotFoundError(f'Input file not found: {src}')

    arr = np.load(src)
    # derive 1D base signal
    if arr.ndim == 2:
        sig = arr.mean(axis=0).ravel()
    else:
        sig = arr.ravel()

    sig_res = _resample_to_length(sig, args.target_length)
    # apply bandpass to make ECG-like frequency content (assume fs ~= 300 Hz)
    sig_res = _bandpass(sig_res, fs=300.0, low=0.5, high=40.0)

    N = args.n_samples
    # create NSR: relatively clean
    nsr_variants = _make_variants(
        sig_res,
        N,
        noise_scale=0.02,
        jitter_shift=30,
        seed=args.seed,
        af=False,
    )

    # create AF: more irregular
    af_variants = _make_variants(
        sig_res,
        N,
        noise_scale=0.04,
        jitter_shift=80,
        seed=(None if args.seed is None else args.seed + 1),
        af=True,
    )

    # expand channel dim and ensure dtype float32
    nsr_out = nsr_variants.reshape((N, args.target_length, 1)).astype(np.float32)
    af_out = af_variants.reshape((N, args.target_length, 1)).astype(np.float32)

    out_af = Path('sssd_article_AF.npy')
    out_nsr = Path('sssd_article_NSR.npy')

    # backup existing
    _backup(out_af)
    _backup(out_nsr)

    np.save(out_af, af_out)
    np.save(out_nsr, nsr_out)

    print('Saved', out_af, 'shape=', af_out.shape)
    print('Saved', out_nsr, 'shape=', nsr_out.shape)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=str, default='SSSD-ECG/SSSD-ECG-main/clinical evaluation/diagnosis on normal samples/data/sample_sssd_norm.npy')
    p.add_argument('--target-length', type=int, default=3000)
    p.add_argument('--n-samples', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    main(args)
