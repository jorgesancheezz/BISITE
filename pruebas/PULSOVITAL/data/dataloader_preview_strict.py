import argparse
import numpy as np
import scipy as sp
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

# ===== COPIED VERBATIM FROM train.py =====

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

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        samples_per_rank = self.num_samples // world_size
        start_idx = rank * samples_per_rank
        end_idx = start_idx + samples_per_rank

        if worker_info is not None:
            samples_per_worker = (end_idx - start_idx) // worker_info.num_workers
            start_idx = start_idx + worker_info.id * samples_per_worker
            end_idx = start_idx + samples_per_worker

        num_samples = end_idx - start_idx

        seed = torch.initial_seed() + start_idx
        np.random.seed(seed % (2**32 - 1))

        datagen = self.generator(
            length=self.length,
            noise_scale=self.noise_scale,
            scale=self.scale,
            **self.generator_kwargs,
        )
        return datagen(num_samples, self.alpha)

# ===== END COPIED PARTS =====


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
