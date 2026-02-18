import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# Reuse existing generators for consistency (prefer core implementation)
try:
    from PULSOVITAL.core.fid_all_in_one import (
        SineDataGenerator,
        WaveletDataGenerator,
        SyntheticIterableDataset,
        ensure_dir,
    )
except Exception:
    # Fallback to optional root shim if present
    from PULSOVITAL.fid_all_in_one import (
        SineDataGenerator,
        WaveletDataGenerator,
        SyntheticIterableDataset,
        ensure_dir,
    )

def build_synthetic_loader(generator: str, length: int, alpha: float, noise: float,
                           batch_size: int = 4, num_samples: int = 128, num_workers: int = 0,
                           scale: float | None = None) -> DataLoader:
    gens = {"sine": SineDataGenerator, "wavelet": WaveletDataGenerator}
    if generator not in gens:
        raise ValueError(f"generator must be one of {list(gens.keys())}, got: {generator}")
    dataset = SyntheticIterableDataset(
        gens[generator],
        num_samples=num_samples,
        length=length,
        alpha=alpha,
        noise_scale=noise,
        scale=scale,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader


def collect_samples(loader: DataLoader, max_samples: int) -> np.ndarray:
    xs = []
    total = 0
    for xb in loader:
        take = min(xb.shape[0], max_samples - total)
        if take <= 0:
            break
        xs.append(xb[:take].to(dtype=torch.float32).cpu().numpy())
        total += take
        if total >= max_samples:
            break
    if not xs:
        return np.empty((0, 0, 0), dtype=np.float32)
    return np.concatenate(xs, axis=0)


def main():
    p = argparse.ArgumentParser(description="Minimal synthetic DataLoader: generator, length, alpha, noise, [scale]")
    p.add_argument("--generator", choices=["sine", "wavelet"], required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--noise", type=float, required=True)
    # optional controls
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--scale", type=float, default=0.1260, help="Scale divisor for amplitude (x/scale). E.g., 0.1260 -> ~7.94x")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    loader = build_synthetic_loader(
        generator=args.generator,
        length=args.length,
        alpha=args.alpha,
        noise=args.noise,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        scale=args.scale,
    )

    arr = collect_samples(loader, args.num_samples)
    print(f"Collected samples: shape={arr.shape} | scale={args.scale}")
    if args.out:
        ensure_dir(args.out)
        np.save(args.out, arr.astype(np.float32))
        print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
