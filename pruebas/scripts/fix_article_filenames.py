import os,shutil
import numpy as np

pairs = [
    ('PULSOVITAL/Metricas/sssd_article_AF_proc_proc_3000.npy','PULSOVITAL/Metricas/sssd_article_AF_proc_3000.npy'),
    ('PULSOVITAL/Metricas/sssd_article_NSR_proc_proc_3000.npy','PULSOVITAL/Metricas/sssd_article_NSR_proc_3000.npy'),
]

for src, dst in pairs:
    if os.path.exists(src):
        # backup existing dst if present
        if os.path.exists(dst):
            bak = dst + '.bak'
            print('Backing up', dst, '->', bak)
            shutil.copy2(dst, bak)
        print('Copying', src, '->', dst)
        shutil.copy2(src, dst)
    else:
        print('Source not found:', src)

# Print resulting shapes
for _, dst in pairs:
    print('---', dst)
    if not os.path.exists(dst):
        print(' MISSING')
        continue
    a = np.load(dst)
    print(' shape:', a.shape, 'dtype:', a.dtype, 'size:', a.size)
