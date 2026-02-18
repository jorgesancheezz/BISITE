import numpy as np
files = [
    'PULSOVITAL/Metricas/sssd_article_AF_proc_proc_3000.npy',
    'PULSOVITAL/Metricas/sssd_article_NSR_proc_proc_3000.npy'
]
for f in files:
    a = np.load(f)
    print(f, a.shape, a.dtype)
