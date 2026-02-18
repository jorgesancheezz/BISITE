import os
from pathlib import Path
import numpy as np
import argparse

# Import WaveletDataGenerator from your PULSOVITAL training module
try:
    from PULSOVITAL.training.train import WaveletDataGenerator
except Exception:
    WaveletDataGenerator = None
    # fallback: load module from file path
    try:
        import importlib.util, sys
        mod_path = str(Path(__file__).resolve().parent.parent / 'PULSOVITAL' / 'training' / 'train.py')
        spec = importlib.util.spec_from_file_location('pulso_train_module', mod_path)
        pulso_mod = importlib.util.module_from_spec(spec)
        sys.modules['pulso_train_module'] = pulso_mod
        spec.loader.exec_module(pulso_mod)
        WaveletDataGenerator = getattr(pulso_mod, 'WaveletDataGenerator', None)
    except Exception:
        WaveletDataGenerator = None


def generate_and_save(out_dir, n_samples=16, length=3000, alpha=0.5, noise_scale=0.0, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    if WaveletDataGenerator is None:
        raise RuntimeError('Could not import WaveletDataGenerator from PULSOVITAL.training.train')
    np.random.seed(seed)
    gen = WaveletDataGenerator(length=length, noise_scale=noise_scale)
    arr = np.zeros((n_samples, length, 1), dtype=np.float32)
    i = 0
    for s in gen(n_samples, alpha):
        # s is a torch tensor with shape (length,1)
        try:
            import torch
            if isinstance(s, torch.Tensor):
                s = s.cpu().numpy()
        except Exception:
            pass
        arr[i, :, 0] = np.asarray(s).reshape(-1)[:length]
        i += 1
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='PULSOVITAL/npy_output_article_converted')
    parser.add_argument('--n', type=int, default=16)
    parser.add_argument('--length', type=int, default=3000)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', type=str, default='sample_wavelet')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = generate_and_save(out_dir, n_samples=args.n, length=args.length, alpha=args.alpha, noise_scale=args.noise, seed=args.seed)
    outp = out_dir / f'{args.name}_{args.n}.npy'
    np.save(outp, arr)
    print('Saved', outp)


if __name__ == '__main__':
    main()
