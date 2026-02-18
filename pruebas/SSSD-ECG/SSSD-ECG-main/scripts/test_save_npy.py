import sys
import os
# allow importing inference.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'sssd')))
from inference import ensure_shape_npy
import numpy as np

# create dummy generated_audio12 with shape (4,8,1000) -> typical model output
dummy = np.random.randn(4,8,1000).astype(np.float32)
arr = ensure_shape_npy(dummy, target=(1024,3000,1), lead_idx=0)
np.save(os.path.join(os.path.dirname(__file__), 'test_1024_3000_1.npy'), arr)
print('saved test file at scripts/test_1024_3000_1.npy, shape =', arr.shape)
