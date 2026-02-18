#!/usr/bin/env python3
import os
import numpy as np

files = ['sssd_article_AF.npy','sssd_article_NSR.npy']
for p in files:
    if os.path.exists(p):
        a = np.load(p)
        print(p)
        print('  path:', os.path.abspath(p))
        print('  shape:', a.shape)
        print('  dtype:', a.dtype)
        print('  size(bytes):', a.nbytes)
        print('  NaNs:', int(np.isnan(a).sum()))
    else:
        print('Missing', p)
