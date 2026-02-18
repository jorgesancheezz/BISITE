import os
import numpy as np
import importlib.util

# load generator from PULSOVITAL/core/fid_all_in_one.py
script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PULSOVITAL', 'core', 'fid_all_in_one.py'))
spec = importlib.util.spec_from_file_location('fid_all_in_one', script_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

WaveletDataGenerator = mod.WaveletDataGenerator

def generate(n_records=22, n_leads=12, length=1000, alpha=0.5, noise=0.0, seed=42, out_path=None):
    rng = np.random.RandomState(seed)
    total = n_records * n_leads
    gen = WaveletDataGenerator(length=length, noise_scale=noise)
    samples = []
    # Use numpy RNG to set global seed for reproducibility in generator which uses np.random
    np.random.seed(seed)
    for i in range(total):
        x = gen.sample(alpha)
        xa = x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)
        # ensure shape (length, 1)
        xa = xa.reshape(length, 1)
        samples.append(xa)
    arr = np.stack(samples, axis=0)  # (total, length, 1)
    arr = arr.reshape(n_records, n_leads, length, 1)
    # Save as (N, leads, T)
    arr_out = arr[:, :, :, 0].astype(np.float32)  # shape (N, leads, T)
    if out_path is None:
        out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'samples_mygen_22.npy'))
    np.save(out_path, arr_out)
    print('Wrote', out_path, 'shape=', arr_out.shape)
    return out_path


if __name__ == '__main__':
    generate()
