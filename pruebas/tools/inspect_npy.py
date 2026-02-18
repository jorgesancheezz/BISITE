import numpy as np, os, sys, time
p = sys.argv[1] if len(sys.argv)>1 else '1024seq_AF.orig.npy'
if not os.path.exists(p):
    print('MISSING', p)
    sys.exit(1)
a = np.load(p)
print('path:', os.path.abspath(p))
print('shape:', a.shape)
print('dtype:', a.dtype)
print('min,max,mean,std:', float(a.min()), float(a.max()), float(a.mean()), float(a.std()))
print('first10 (flatten):', a.ravel()[:10])
print('mtime:', time.ctime(os.path.getmtime(p)))
