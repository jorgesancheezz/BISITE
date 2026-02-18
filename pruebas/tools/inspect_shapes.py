import numpy as np
files = [
    r'C:/Users/BISITE-NEL/Desktop/pruebas/samples_real_22.npy',
    r'C:/Users/BISITE-NEL/Desktop/pruebas/samples_synthetic_22.npy',
    r'C:/Users/BISITE-NEL/Desktop/pruebas/1024seq_AF.npy',
    r'C:/Users/BISITE-NEL/Desktop/pruebas/1024seq_NSR.npy'
]
for f in files:
    try:
        a = np.load(f)
        print(f, 'shape=', a.shape, 'dtype=', a.dtype)
    except Exception as e:
        print(f, 'ERROR', e)
