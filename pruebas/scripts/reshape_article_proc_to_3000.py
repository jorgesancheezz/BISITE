import os
from math import gcd
from scipy.signal import resample_poly
import numpy as np

FILES = [
    "PULSOVITAL/Metricas/sssd_article_AF_proc.npy",
    "PULSOVITAL/Metricas/sssd_article_NSR_proc.npy",
]
TARGET_LEN = 3000

for p in FILES:
    if not os.path.exists(p):
        print(f"Not found: {p}")
        continue
    arr = np.load(p)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected ndim for {p}: {arr.ndim}")
    n, L, c = arr.shape
    if L == TARGET_LEN:
        out_p = p.replace('.npy', '_proc_3000.npy')
        np.save(out_p, arr.astype(np.float16))
        print(f"Already target length, copied -> {out_p}")
        continue
    up = TARGET_LEN
    down = L
    g = gcd(up, down)
    up //= g
    down //= g
    print(f"Resampling {p}: {L} -> {TARGET_LEN} (up={up}, down={down})")
    try:
        res = resample_poly(arr, up, down, axis=1)
    except Exception as e:
        print("resample_poly failed, falling back to interp:", e)
        x_old = np.linspace(0, 1, L)
        x_new = np.linspace(0, 1, TARGET_LEN)
        res = np.stack([np.interp(x_new, x_old, arr[i, :, 0]) for i in range(n)], axis=0)
        res = res[:, :, None]
    if res.shape[1] != TARGET_LEN:
        res = res[:, :TARGET_LEN, :]
    out_p = p.replace('.npy', '_proc_3000.npy')
    np.save(out_p, res.astype(np.float16))
    print(f"Saved {out_p} shape={res.shape} dtype={res.dtype}")
