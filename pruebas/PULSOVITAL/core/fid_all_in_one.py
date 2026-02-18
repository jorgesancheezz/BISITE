
import argparse
import os
from typing import Tuple

import numpy as np
import scipy as sp
from scipy import linalg
from scipy.signal import spectrogram
import torch
from torch.utils.data import DataLoader, IterableDataset

#########################################################################################
# Parts copied from train.py (unaltered APIs)
#########################################################################################

class SineDataGenerator:
    def __init__(
        self,
        length,
        *,
        min_low_amplitude=0.1,
        max_low_amplitude=1.0,
        min_high_amplitude=1.0,
        max_high_amplitude=15.0,
        min_cycle_length=3,
        max_cycle_length=10,
        min_frequency=1.0,
        max_frequency=5.0,
        num_periods=25.0,
        noise_scale=0.0,
        scale=None,
    ):
        self.length = length
        self.min_low_amplitude = min_low_amplitude
        self.max_low_amplitude = max_low_amplitude
        self.min_high_amplitude = min_high_amplitude
        self.max_high_amplitude = max_high_amplitude
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.noise_scale = noise_scale
        self.scale = scale

        max_period_length = np.ceil(2.0 * np.pi / self.min_frequency)
        self.t = np.linspace(
            0.0,
            2.0 * np.pi * num_periods * ((self.length + max_period_length) / self.length),
            int(self.length + max_period_length),
        )

    def sample(self, alpha):
        low_amplitude = self.max_low_amplitude + alpha * (
            self.min_low_amplitude - self.max_low_amplitude
        )
        high_amplitude = self.min_high_amplitude + alpha * (
            self.max_high_amplitude - self.min_high_amplitude
        )

        cycle_length = round(
            self.min_cycle_length + np.random.rand() * (self.max_cycle_length - self.min_cycle_length)
        )
        frequency = self.min_frequency + np.random.rand() * (self.max_frequency - self.min_frequency)

        period_length = 2.0 * np.pi / frequency
        initial_phase = np.random.randint(low=0, high=int(np.ceil(period_length)))
        cycle_indices = np.floor(self.t / period_length) + 1
        starting_beat = np.random.randint(low=1, high=cycle_length + 1)

        low_sine = low_amplitude * np.sin(frequency * self.t)
        high_sine = high_amplitude * np.sin(frequency * self.t)

        x = np.where((cycle_indices - starting_beat) % cycle_length == 0, high_sine, low_sine)
        x = x[initial_phase : initial_phase + self.length]

        x = (
            torch.from_numpy(x).unsqueeze(-1)
            if self.scale is None
            else torch.from_numpy(x).unsqueeze(-1) / self.scale
        )
        x = x + self.noise_scale * torch.randn_like(x) if self.noise_scale > 0.0 else x

        return x

    def __call__(self, n, alpha):
        for _ in range(n):
            yield self.sample(alpha)


class WaveletDataGenerator:
    def __init__(
        self,
        length,
        *,
        base_spacing=15.0,
        spacing_variability=0.3,
        min_width=0.025,
        max_width=0.25,
        width_variability=0.2,
        base_frequency=10.0,
        frequency_variability=0.2,
        total_time=50.0,
        noise_scale=0.0,
        scale=None,
    ):
        self.length = length
        self.base_spacing = base_spacing
        self.spacing_variability = spacing_variability
        self.min_width = min_width * base_spacing
        self.max_width = max_width * base_spacing
        self.width_variability = width_variability
        self.base_frequency = base_frequency
        self.frequency_variability = frequency_variability
        self.noise_scale = noise_scale
        self.scale = scale

        self.extra = int(np.ceil((self.base_spacing / total_time) * self.length))
        self.t = np.linspace(0.0, total_time * (1.0 + (self.extra / self.length)), self.length + self.extra)

    def sample(self, alpha):
        offset = np.random.randint(low=0, high=self.extra)
        frequency = self.base_frequency * (1 + self.frequency_variability * (np.random.rand() - 0.5) * 2)

        last_center = -self.t.max()
        centers = []
        widths = []
        done = False

        while not done:
            new_center = last_center + self.base_spacing * (1 + self.spacing_variability * (np.random.rand() - 0.5) * 2)
            if new_center <= 2.0 * self.t.max():
                centers.append(new_center)
                last_center = new_center

                new_width = (self.max_width + alpha * (self.min_width - self.max_width)) * (
                    1 + self.width_variability * (np.random.rand() - 0.5) * 2
                )
                widths.append(new_width)
            else:
                done = True

        wavelets = np.zeros_like(self.t)
        for c, w in zip(centers, widths):
            wavelets = wavelets + sp.stats.norm.pdf(self.t, c, w)

        x = wavelets * np.sin(frequency * self.t)
        x = x[offset : offset + self.length]

        x = (
            torch.from_numpy(x).unsqueeze(-1)
            if self.scale is None
            else torch.from_numpy(x).unsqueeze(-1) / self.scale
        )
        x = x + self.noise_scale * torch.randn_like(x) if self.noise_scale > 0.0 else x

        return x

    def __call__(self, n, alpha):
        for _ in range(n):
            yield self.sample(alpha)


class SyntheticIterableDataset(IterableDataset):
    def __init__(
        self,
        generator,
        /,
        num_samples,
        length,
        alpha,
        *,
        noise_scale=0.0,
        scale=None,
        **kwargs,
    ):
        self.generator = generator
        self.num_samples = num_samples
        self.length = length
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.scale = scale
        self.generator_kwargs = kwargs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # Split work only across DataLoader workers (no distributed setup)
        if worker_info is not None and worker_info.num_workers > 0:
            samples_per_worker = max(self.num_samples // worker_info.num_workers, 1)
            num_samples = samples_per_worker
            seed_offset = worker_info.id
        else:
            num_samples = self.num_samples
            seed_offset = 0

        seed = torch.initial_seed() + seed_offset
        np.random.seed(seed % (2**32 - 1))

        datagen = self.generator(
            length=self.length,
            noise_scale=self.noise_scale,
            scale=self.scale,
            **self.generator_kwargs,
        )
        return datagen(num_samples, self.alpha)

# ===== END COPIED PARTS =====


#############################################
# Helper utilities expected by metrics module
#############################################

def parse_kwargs(seq) -> dict:
    """Parse a flat sequence like [k1, v1, k2, v2, ...] into a dict of floats.
    If seq is None or empty, returns {}.
    """
    if not seq:
        return {}
    it = iter(seq)
    out = {}
    for k in it:
        try:
            v = next(it)
        except StopIteration:
            break
        try:
            out[str(k)] = float(v)
        except Exception:
            # keep raw if not float-castable
            out[str(k)] = v
    return out


def frange(start: float, stop: float, step: float):
    """Generate a list of floats from start to stop (inclusive-ish) with step.
    Uses a guard to include the endpoint when numerically close.
    """
    vals = []
    x = float(start)
    step = float(step)
    stop = float(stop)
    # support both ascending and descending ranges
    if step == 0:
        return [x]
    if (step > 0 and x > stop) or (step < 0 and x < stop):
        return []
    while (step > 0 and x <= stop + 1e-12) or (step < 0 and x >= stop - 1e-12):
        vals.append(x)
        x += step
    return vals


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def compute_stats(arr: np.ndarray):
    """Return (mean, covariance) for 2D array [N, D]."""
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((arr.shape[-1] if arr.ndim == 2 else 0,), dtype=np.float64), np.eye(
            arr.shape[-1] if arr.ndim == 2 else 1, dtype=np.float64
        )
    mu = np.mean(arr, axis=0)
    sigma = np.cov(arr, rowvar=False)
    if sigma.ndim == 0:
        sigma = np.array([[float(sigma)]], dtype=np.float64)
    return mu, sigma


def _extract_batch_features(xb: torch.Tensor) -> np.ndarray:
    """Compute simple spectrogram features per item in a batch.
    Input xb: torch tensor of shape (B, T, 1).
    Returns np.ndarray (B, F) of log-mean spectrogram magnitudes across time.
    """
    xb = xb.detach().cpu().numpy()  # (B, T, 1)
    B, T, C = xb.shape
    feats = []
    for i in range(B):
        x = xb[i, :, 0]
        f, t, Sxx = spectrogram(x, nperseg=256, noverlap=128, detrend=False, scaling="density")
        m = np.mean(np.log(Sxx + 1e-12), axis=1)
        feats.append(m.astype(np.float32))
    return np.stack(feats, axis=0)

# Back-compat export name expected by metrics.compare_stats
extract_batch_features = _extract_batch_features


@torch.no_grad()
def collect_features(loader: DataLoader, max_samples: int, device: torch.device) -> np.ndarray:
    all_feats = []
    total = 0
    for xb in loader:
        xb = xb.to(dtype=torch.float32, device=device)
        feats = _extract_batch_features(xb)
        all_feats.append(feats)
        total += feats.shape[0]
        if total >= max_samples:
            break
    if not all_feats:
        return np.empty((0, 0), dtype=np.float32)
    out = np.concatenate(all_feats, axis=0).astype(np.float32)
    return out[:max_samples]


def build_loader(
    generator_name: str,
    length: int,
    alpha: float,
    noise: float,
    batch_size: int,
    samples: int,
    num_workers: int,
    generator_kwargs: dict | None = None,
) -> DataLoader:
    gens = {"sine": SineDataGenerator, "wavelet": WaveletDataGenerator}
    Gen = gens[generator_name]
    generator_kwargs = generator_kwargs or {}
    dataset = SyntheticIterableDataset(
        Gen,
        num_samples=samples,
        length=length,
        alpha=alpha,
        noise_scale=noise,
        **generator_kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Standard Frechet distance between two Gaussians (mu, sigma)."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    def _sqrtm_compat(A: np.ndarray) -> np.ndarray:
        res = linalg.sqrtm(A)
        return res[0] if isinstance(res, tuple) else res

    diff = mu1 - mu2
    covmean = _sqrtm_compat(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _sqrtm_compat((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    return fid


def main():
    _generators = {"sine": SineDataGenerator, "wavelet": WaveletDataGenerator}
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", choices=list(_generators.keys()), default="wavelet")
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--generator-kwargs", nargs="+", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=1_000_000)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    generator = _generators[args.generator]
    generator_kwargs_iter = iter(args.generator_kwargs)
    generator_kwargs = {k: float(v) for k, v in dict(zip(generator_kwargs_iter, generator_kwargs_iter)).items()}

    dataset = SyntheticIterableDataset(
        generator,
        num_samples=args.train_samples,
        length=args.length,
        alpha=args.alpha,
        noise_scale=args.noise,
        **generator_kwargs,
    )

    # Exactly the same DataLoader flags as in train.py
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # No extra logic: DataLoader is constructed here as in train.py.
    # If you want to iterate, just do: for x in loader: break

if __name__ == "__main__":
    main()
