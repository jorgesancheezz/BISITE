import numpy as np
from pathlib import Path

src = Path('SSSD-ECG/SSSD-ECG-main/clinical evaluation/diagnosis on normal samples/data/sample_sssd_norm.npy')
out_af = Path('sssd_article_AF.npy')
out_nsr = Path('sssd_article_NSR.npy')

x = np.load(src)  # shape (12,1000)
if x.ndim == 2:
    sig = x.mean(axis=0)
else:
    sig = x.ravel()

T_orig = sig.shape[0]
T_target = 3000
xp = np.arange(T_orig)
x_new = np.linspace(0, T_orig - 1, T_target)
sig_res = np.interp(x_new, xp, sig).astype(np.float32)

N = 1024
arr = np.tile(sig_res, (N, 1)).reshape((N, T_target, 1)).astype(np.float32)
np.save(out_af, arr)
np.save(out_nsr, arr)
print('Saved', out_af, 'and', out_nsr, 'shape=', arr.shape)
