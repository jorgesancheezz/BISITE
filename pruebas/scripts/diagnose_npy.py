import os, sys
import numpy as np

candidates = [
    'PULSOVITAL/Metricas/sssd_article_AF_proc_proc_3000.npy',
    'PULSOVITAL/Metricas/sssd_article_NSR_proc_proc_3000.npy',
    'PULSOVITAL/Metricas/sssd_article_AF_proc_3000.npy',
    'PULSOVITAL/Metricas/sssd_article_NSR_proc_3000.npy',
    'PULSOVITAL/Metricas/sssd_article_AF_proc.npy',
    'PULSOVITAL/Metricas/sssd_article_NSR_proc.npy',
    'PULSOVITAL/Metricas/sssd_article_AF.npy',
    'PULSOVITAL/Metricas/sssd_article_NSR.npy',
    '1024seq_AF.npy',
    '1024seq_NSR.npy',
    'AF_processed_1024x3000x1.npy',
]

os.makedirs('notebooks/diagnostic_outputs', exist_ok=True)

for p in candidates:
    print('---')
    print('File:', p)
    if not os.path.exists(p):
        print('  EXISTS: False')
        continue
    print('  EXISTS: True')
    try:
        a = np.load(p)
    except Exception as e:
        print('  LOAD ERROR:', e)
        continue
    print('  shape:', getattr(a, 'shape', None), 'dtype:', getattr(a, 'dtype', None))
    try:
        size = a.size
    except Exception:
        size = None
    print('  size:', size)
    try:
        flat = np.ravel(a.astype('float64'))
        nancount = int(np.isnan(flat).sum())
        infcount = int(np.isinf(flat).sum())
        print('  NaNs:', nancount, 'Inf:', infcount)
        if flat.size>0:
            print('  min/max/mean:', float(np.nanmin(flat)), float(np.nanmax(flat)), float(np.nanmean(flat)))
    except Exception as e:
        print('  stats error:', e)
    # if 3D and last dim==1, show first sample summary
    try:
        arr = a
        if getattr(arr,'ndim',0) == 3 and arr.shape[2]==1:
            s0 = arr[0,:,0]
            print('  sample0 mean/std/len:', float(np.mean(s0)), float(np.std(s0)), len(s0))
        elif getattr(arr,'ndim',0) >= 1:
            s0 = arr[0]
            print('  sample0 mean/std/len:', float(np.mean(s0)), float(np.std(s0)), getattr(s0,'shape',None))
    except Exception as e:
        print('  sample summary error:', e)

print('\nDiagnostic finished.')
