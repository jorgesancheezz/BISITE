import argparse
import os
from typing import List
import numpy as np
import torch

try:
    from PULSOVITAL.fid_all_in_one import WaveletDataGenerator, ensure_dir
except Exception:
    import sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(__file__))
    if _root not in sys.path:
        sys.path.append(_root)
    from PULSOVITAL.fid_all_in_one import WaveletDataGenerator, ensure_dir


def save_npy(out_path: str, arr: np.ndarray):
    ensure_dir(out_path)
    np.save(out_path, arr)
    print(f"Saved .npy array to {out_path} with shape {arr.shape}")


def save_csv(out_path: str, arr: np.ndarray):
    ensure_dir(out_path)
    # Flatten last dim if it's 1
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    # Save as rows: each sample is a row, columns are time steps
    np.savetxt(out_path, arr.astype(np.float32), delimiter=",")
    print(f"Saved .csv matrix to {out_path} with shape {arr.shape}")


def save_shard(base_out: str, fmt: str, shard_idx: int, data: np.ndarray) -> str:
    root, ext = os.path.splitext(base_out)
    if fmt == "npy":
        path = f"{root}_shard{shard_idx:05d}.npy"
        save_npy(path, data)
    else:
        path = f"{root}_shard{shard_idx:05d}.csv"
        save_csv(path, data)
    return path


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic wavelet data with given alpha and noise.")
    ap.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    ap.add_argument("--length", type=int, default=1000, help="Signal length (timesteps)")
    ap.add_argument("--alpha", type=float, default=0.9, help="Wavelet generator alpha in [0,1]")
    ap.add_argument("--noise", type=float, default=0.05, help="Gaussian noise scale")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--format", choices=["npy", "csv"], default="npy", help="Output format")
    ap.add_argument("--out", type=str, default=os.path.join("PULSOVITAL", "results", "wavelet_alpha_data.npy"), help="Output file path")
    # Advanced options
    ap.add_argument("--patch-sizes", type=int, nargs="*", default=None, help="If provided, extract random patches of these sizes from each generated signal and save separately")
    ap.add_argument("--shard-size", type=int, default=1000, help="Number of samples per output shard file (for large datasets)")
    ap.add_argument("--scale", type=float, default=None, help="Scale divisor for amplitude (x/scale). For example scale=0.1260 amplifies ~7.94x")
    args = ap.parse_args()

    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)

    gen = WaveletDataGenerator(length=args.length, noise_scale=args.noise, scale=args.scale)

    if not args.patch_sizes:
        # Simple mode: save full signals, optionally sharded
        expected_shards = max(1, (args.samples + args.shard_size - 1) // args.shard_size)
        shard = []
        shard_idx = 0
        for i in range(args.samples):
            x = gen.sample(args.alpha).unsqueeze(0)  # [1,T,1]
            shard.append(x)
            if len(shard) >= args.shard_size:
                data = torch.cat(shard, dim=0).to(dtype=torch.float32).numpy()
                # If only one shard is expected, save without suffix
                if expected_shards == 1 and shard_idx == 0:
                    if args.format == "npy":
                        save_npy(args.out, data)
                    else:
                        save_csv(args.out, data)
                else:
                    save_shard(args.out, args.format, shard_idx, data)
                shard_idx += 1
                shard = []
        if shard:
            data = torch.cat(shard, dim=0).to(dtype=torch.float32).numpy()
            if expected_shards == 1 and shard_idx == 0:
                if args.format == "npy":
                    save_npy(args.out, data)
                else:
                    save_csv(args.out, data)
            else:
                save_shard(args.out, args.format, shard_idx, data)
    else:
        # Patch mode: create separate outputs per patch size
        patch_sizes: List[int] = list(args.patch_sizes)
        for p in patch_sizes:
            assert p > 0 and p <= args.length, f"Invalid patch size {p} for length {args.length}"
            base_out = args.out
            # put in a subpath with patch size
            root, ext = os.path.splitext(base_out)
            base_out = f"{root}_patch{p}{ext if ext else ('.npy' if args.format=='npy' else '.csv')}"
            expected_shards = max(1, (args.samples + args.shard_size - 1) // args.shard_size)
            shard = []
            shard_idx = 0
            for i in range(args.samples):
                x = gen.sample(args.alpha)  # [T,1]
                # random crop of size p
                if args.length == p:
                    xp = x
                else:
                    start = np.random.randint(0, args.length - p + 1)
                    xp = x[start:start+p]
                shard.append(xp.unsqueeze(0))  # [1,p,1]
                if len(shard) >= args.shard_size:
                    data = torch.cat(shard, dim=0).to(dtype=torch.float32).numpy()
                    if expected_shards == 1 and shard_idx == 0:
                        if args.format == "npy":
                            save_npy(base_out, data)
                        else:
                            save_csv(base_out, data)
                    else:
                        save_shard(base_out, args.format, shard_idx, data)
                    shard_idx += 1
                    shard = []
            if shard:
                data = torch.cat(shard, dim=0).to(dtype=torch.float32).numpy()
                if expected_shards == 1 and shard_idx == 0:
                    if args.format == "npy":
                        save_npy(base_out, data)
                    else:
                        save_csv(base_out, data)
                else:
                    save_shard(base_out, args.format, shard_idx, data)


if __name__ == "__main__":
    main()
