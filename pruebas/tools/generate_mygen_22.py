import os
import numpy as np
import importlib.util
import torch

# Load WaveletDataGenerator from PULSOVITAL/core/fid_all_in_one.py
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PULSOVITAL', 'core', 'fid_all_in_one.py'))
spec = importlib.util.spec_from_file_location('fid_all_in_one', root)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
WaveletDataGenerator = mod.WaveletDataGenerator


def generate(n_samples=22, n_leads=12, length=1000, alpha=0.9, noise=0.05, seed=42, out_path=None):
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    gen = WaveletDataGenerator(length=length, noise_scale=noise)
    out = np.zeros((n_samples, n_leads, length), dtype=np.float32)
    for i in range(n_samples):
        for l in range(n_leads):
            x = gen.sample(alpha)  # torch tensor [T,1]
            x_np = x.detach().cpu().numpy().reshape(-1)
            out[i, l, :] = x_np
    if out_path is None:
        out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'samples_mygen_22.npy'))
    np.save(out_path, out)
    print(f"Saved generated samples to {out_path} with shape {out.shape}")
    return out_path


if __name__ == '__main__':
    generate()
