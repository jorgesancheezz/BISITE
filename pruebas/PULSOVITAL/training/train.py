# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""

#########################################################################################
# region                                 IMPORTS                                        #
#########################################################################################

import argparse
from collections import OrderedDict
from copy import deepcopy
import os
from time import time

import numpy as np
import scipy as sp
try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception as _e_sklearn:  # pragma: no cover
    StandardScaler = None  # type: ignore
    _SKLEARN_AVAILABLE = False
    _SKLEARN_IMPORT_ERROR = _e_sklearn
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except Exception as _e_yaml:  # pragma: no cover
    yaml = None  # type: ignore
    _YAML_AVAILABLE = False
    _YAML_IMPORT_ERROR = _e_yaml

# Some training utilities depend on an external 'diffusion' package. Make import optional
# so that merely importing this module or running "-m ... --help" doesn't crash.
try:
    from diffusion.aux import create_logger  # type: ignore
    from diffusion.gaussian_diffusion import GaussianDiffusion  # type: ignore
    from diffusion.models import DiT_models  # type: ignore
    _DIFFUSION_AVAILABLE = True
except Exception as _e:  # pragma: no cover - optional dependency guard
    create_logger = None  # type: ignore
    GaussianDiffusion = None  # type: ignore
    DiT_models = None  # type: ignore
    _DIFFUSION_AVAILABLE = False
    _DIFFUSION_IMPORT_ERROR = _e

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# endregion

#########################################################################################
# region                            SYNTHETIC DATASET                                   #
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
            2.0
            * np.pi
            * num_periods
            * ((self.length + max_period_length) / self.length),
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
            self.min_cycle_length
            + np.random.rand() * (self.max_cycle_length - self.min_cycle_length)
        )
        frequency = self.min_frequency + np.random.rand() * (
            self.max_frequency - self.min_frequency
        )

        period_length = 2.0 * np.pi / frequency  # + 1e-6
        initial_phase = np.random.randint(low=0, high=int(np.ceil(period_length)))
        cycle_indices = np.floor(self.t / period_length) + 1
        starting_beat = np.random.randint(low=1, high=cycle_length + 1)

        low_sine = low_amplitude * np.sin(frequency * self.t)
        high_sine = high_amplitude * np.sin(frequency * self.t)

        x = np.where(
            (cycle_indices - starting_beat) % cycle_length == 0, high_sine, low_sine
        )
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
        self.t = np.linspace(
            0.0,
            total_time * (1.0 + (self.extra / self.length)),
            self.length + self.extra,
        )

    def sample(self, alpha):
        offset = np.random.randint(low=0, high=self.extra)
        frequency = self.base_frequency * (
            1 + self.frequency_variability * (np.random.rand() - 0.5) * 2
        )

        last_center = -self.t.max()
        centers = []
        widths = []
        done = False

        while not done:
            new_center = last_center + self.base_spacing * (
                1 + self.spacing_variability * (np.random.rand() - 0.5) * 2
            )
            if new_center <= 2.0 * self.t.max():
                centers.append(new_center)
                last_center = new_center

                new_width = (
                    self.max_width + alpha * (self.min_width - self.max_width)
                ) * (1 + self.width_variability * (np.random.rand() - 0.5) * 2)
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
        # sharding
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


# endregion

#########################################################################################
# region                        TRAINING HELPER FUNCTIONS                               #
#########################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if param.requires_grad:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


# endregion

#########################################################################################
# region                              MAIN FUNCTION                                     #
#########################################################################################


def main(args):
    if not _DIFFUSION_AVAILABLE:
        raise RuntimeError(
            "This training script requires the optional 'diffusion' package. "
            f"Import error was: {_DIFFUSION_IMPORT_ERROR}.\n"
            "You can still use PULSOVITAL.training.similarity_train for a lightweight, \n"
            "self-contained training example that does not require external dependencies."
        )
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    #####################################################################################
    # region                  PREPARE SYNTHETIC DATA GENERATOR                          #
    #####################################################################################

    generators = {"sine": SineDataGenerator, "wavelet": WaveletDataGenerator}
    generator = generators[args.generator]

    generator_kwargs_iter = iter(args.generator_kwargs)
    generator_kwargs = {
        k: float(v)
        for k, v in dict(zip(generator_kwargs_iter, generator_kwargs_iter)).items()
    }
    # endregion

    #####################################################################################
    # region                           INITIALIZE DDP                                   #
    #####################################################################################

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # endregion

    #####################################################################################
    # region               INITIALIZE EXPERIMENT FOLDER AND LOGGER                      #
    #####################################################################################

    if rank == 0:
        os.makedirs(args.results, exist_ok=True)
        expid = (
            max([int(d[:3]) + 1 for d in os.listdir(args.results)])
            if len(os.listdir(args.results)) > 0
            else 1
        )
        model_name = f"DiT-{args.model.upper()}"
        experiment_directory = os.path.abspath(
            os.path.join(args.results, f"{expid:03d}-{model_name}")
        )
        checkpoint_directory = os.path.join(experiment_directory, "checkpoints")
        os.makedirs(checkpoint_directory, exist_ok=True)
        logger = create_logger(experiment_directory, logging_level=args.logger)
        logger.info(f"Experiment directory created at {experiment_directory}")

        if _YAML_AVAILABLE:
            with open(os.path.join(experiment_directory, "args.yaml"), "w") as f:
                yaml.dump(vars(args), f)  # type: ignore
        else:
            logger.warning(
                "PyYAML not available; skipping args.yaml export. Install pyyaml to enable this."
            )

    else:
        logger = create_logger(None)

    # endregion

    #####################################################################################
    # region                          STANDARD SCALING                                  #
    #####################################################################################

    logger.info(f"Scaling with {args.scaler_samples} samples...")

    if rank == 0:
        if args.scaler_samples > 0:
            if not _SKLEARN_AVAILABLE:
                logger.warning(
                    "scikit-learn not available; skipping scaling. Install scikit-learn or set --scaler-samples 0."
                )
                scale = None
            else:
                scaler_dataset = SyntheticIterableDataset(
                    generator,
                    num_samples=args.scaler_samples,
                    length=args.length,
                    alpha=args.alpha,
                    **generator_kwargs,
                )
                scaler_dataloader = DataLoader(scaler_dataset, batch_size=args.batch_size)

                scaler = StandardScaler(with_mean=False)

                for x in scaler_dataloader:
                    scaler.partial_fit(x.reshape(-1, x.shape[-1]))
                scale = torch.from_numpy(scaler.var_**0.5)
        else:
            scale = None

    logger.info(f"Scaling done")

    # endregion

    #####################################################################################
    # region                   INITIALIZE MODEL AND OPTIMIZER                           #
    #####################################################################################

    if rank == 0:
        model = DiT_models[model_name](
            length=args.length,
            channels=1,
        )
        logger.info(
            f"Initialized {model_name} model with {sum(p.numel() for p in model.parameters()):,} parameters"
        )

        schedule_kwargs_iter = iter(args.schedule_kwargs)
        noise_schedule_kwargs = {
            k: float(v)
            for k, v in dict(zip(schedule_kwargs_iter, schedule_kwargs_iter)).items()
        }
        diffusion = GaussianDiffusion(
            noise_schedule=args.schedule,
            target=args.target,
            **noise_schedule_kwargs,
        )
    else:
        pass

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[rank])

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # endregion

    #####################################################################################
    # region                      TRAINING INITIALIZATIONS                              #
    #####################################################################################

    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")

    # endregion

    #####################################################################################
    # region                            TRAINING LOOP                                   #
    #####################################################################################

    for epoch in range(1, args.epochs + 1):
        loader_start_time = time()
        train_dataset = SyntheticIterableDataset(
            generator,
            num_samples=args.train_samples,
            length=args.length,
            noise_scale=args.noise,
            alpha=args.alpha,
            scale=scale,
            **generator_kwargs,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        loader_end_time = time()
        start_time += loader_end_time - loader_start_time
        logger.info(f"Beginning epoch {epoch}...")

        for x in train_loader:
            #############################################################################
            # region                    FORWARD-BACKWARD PASS                           #
            #############################################################################

            x = x.to(dtype=torch.float32, device=device)
            t = torch.rand(x.shape[0], dtype=torch.float32, device=device)
            loss = diffusion.training_losses(model, x, t).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            # endregion

            #############################################################################
            # region                      PROGRESS LOGGING                              #
            #############################################################################

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                logger.info(
                    f"(step={train_steps: 10d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )

                running_loss = 0.0
                log_steps = 0
                start_time = time()

            # endregion

            #############################################################################
            # region                        CHECKPOINTING                               #
            #############################################################################

            if train_steps % args.ckpt_every == 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": {
                            "length": model.module.length,
                            "channels": model.module.channels,
                            **vars(args),
                        },
                    }
                    checkpoint_path = os.path.join(
                        checkpoint_directory, f"{train_steps:010d}.pt"
                    )
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            # endregion

            #############################################################################
            # region                         VALIDATION                                 #
            #############################################################################

            if args.validate_every > 0 and train_steps % args.validate_every == 0:
                torch.cuda.synchronize()
                valid_start_time = time()

                with torch.no_grad():
                    model.eval()
                    valid_running_loss = 0.0
                    valid_steps = 0
                    valid_dataset = SyntheticIterableDataset(
                        generator,
                        num_samples=args.valid_samples,
                        length=args.length,
                        noise_scale=args.noise,
                        alpha=args.alpha,
                        scale=scale,
                        **generator_kwargs,
                    )
                    valid_loader = DataLoader(
                        valid_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True,
                    )

                    for x in valid_loader:
                        x = x.to(dtype=torch.float32, device=device)
                        t = torch.rand(x.shape[0], dtype=torch.float32, device=device)

                        loss = diffusion.training_losses(ema, x, t).mean()

                        valid_running_loss += loss.item()
                        valid_steps += 1

                    avg_valid_loss = valid_running_loss / valid_steps

                    valid_end_time = time()
                    valid_steps_per_sec = valid_steps / (
                        valid_end_time - valid_start_time
                    )
                    logger.info(
                        f"(step={train_steps: 10d}) Validation Loss: {avg_valid_loss:.4f}, Validation Steps/Sec: {valid_steps_per_sec:.2f}"
                    )

                start_time = start_time + (valid_end_time - valid_start_time)
                model.train()
                torch.cuda.synchronize()

            # endregion

    # endregion

    #####################################################################################
    # region                             EVALUATION                                     #
    #####################################################################################

    model.eval()

    # endregion

    logger.info("Done!")
    cleanup()


# endregion

if __name__ == "__main__":
    _loggers = ("debug", "info", "warning", "error", "critical")
    _schedules = ("linear", "cosine", "naive-linear", "naive-cosine", "learned")
    _targets = ("x0", "eps", "v")
    _generators = ("sine", "wavelet")
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument("--generator", choices=_generators, default="wavelet")
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--generator-kwargs", nargs="+", default=[])
    parser.add_argument("--scaler-samples", type=int, default=100_000)
    # Pathing and logging parameters
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--model", type=str, default="xs")
    parser.add_argument("--logger", choices=_loggers, default="info")
    # Diffusion parameters
    parser.add_argument("--schedule", choices=_schedules, default="linear")
    parser.add_argument("--schedule-kwargs", nargs="+", default=[])
    parser.add_argument("--target", choices=_targets, default="eps")
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=1_000_000)
    parser.add_argument("--valid-samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--validate-every", type=int, default=50_000)

    args = parser.parse_args()
    main(args)
