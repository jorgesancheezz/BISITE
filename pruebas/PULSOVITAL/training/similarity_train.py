import argparse
import os
import csv
from typing import Iterator, Tuple, Optional
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader

try:
    from PULSOVITAL.core.fid_all_in_one import SineDataGenerator, WaveletDataGenerator, ensure_dir
except Exception:
    # Fallback to root shim if present
    from PULSOVITAL.fid_all_in_one import SineDataGenerator, WaveletDataGenerator, ensure_dir


# ------------------------------
# Dataset that yields (x1, x2, y)
# ------------------------------
class PairIterableDataset(IterableDataset):
    def __init__(
        self,
        gen_name: str,
        length: int,
        noise: float,
        tau: float,
        steps: int,
        pos_ratio: float = 0.5,
        hard_neg_ratio: float = 0.5,
        hard_margin: float = 0.05,
        seed: int = 0,
    ) -> None:
        gens = {"sine": SineDataGenerator, "wavelet": WaveletDataGenerator}
        if gen_name not in gens:
            raise ValueError(f"Unknown generator: {gen_name}")
        self.Gen = gens[gen_name]
        self.length = int(length)
        self.noise = float(noise)
        self.tau = float(tau)
        self.steps = int(steps)
        self.pos_ratio = float(pos_ratio)
        self.hard_neg_ratio = float(hard_neg_ratio)
        self.hard_margin = float(hard_margin)
        self.seed = int(seed)

    def _sample_alpha_pair(self, rng: np.random.Generator, positive: bool) -> Tuple[float, float]:
        # Sample a1 uniformly and then a2 depending on label and hardness
        a1 = float(rng.uniform(0.0, 1.0))
        if positive:
            # force |a1-a2| <= tau using a truncated uniform window
            low = max(0.0, a1 - self.tau)
            high = min(1.0, a1 + self.tau)
            a2 = float(rng.uniform(low, high))
            return a1, a2
        else:
            # negative: |a1-a2| > tau, optionally near the margin (hard)
            hard = rng.uniform() < self.hard_neg_ratio
            max_tries = 100
            for _ in range(max_tries):
                a2 = float(rng.uniform(0.0, 1.0))
                d = abs(a1 - a2)
                if hard:
                    if self.tau < d <= self.tau + self.hard_margin:
                        return a1, a2
                else:
                    if d > (self.tau + self.hard_margin):
                        return a1, a2
            # fallback
            a2 = float(rng.uniform(0.0, 1.0))
            return a1, a2

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Split by DataLoader workers
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers > 0:
            steps = max(self.steps // worker.num_workers, 1)
            seed_offset = worker.id
        else:
            steps = self.steps
            seed_offset = 0

        rng = np.random.default_rng(self.seed + int(torch.initial_seed()) + seed_offset)
        gen = self.Gen(length=self.length, noise_scale=self.noise)

        for _ in range(steps):
            positive = rng.uniform() < self.pos_ratio
            a1, a2 = self._sample_alpha_pair(rng, positive)
            x1 = gen.sample(a1).to(dtype=torch.float32)  # (T, 1) tensor
            x2 = gen.sample(a2).to(dtype=torch.float32)  # (T, 1) tensor
            y = torch.tensor(1.0 if positive else 0.0, dtype=torch.float32)
            yield x1, x2, y


# ------------------------------
# Simple 1D CNN encoder
# ------------------------------
class CNNEncoder1D(nn.Module):
    def __init__(self, in_channels: int, hidden: int, emb_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.net(x)
        z = self.proj(h)
        z = nn.functional.normalize(z, dim=-1)
        return z


# ------------------------------
# Metrics
# ------------------------------
def auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # Fall back to a simple rank-based AUC computation
        y_true = y_true.astype(np.float64).reshape(-1)
        y_score = y_score.astype(np.float64).reshape(-1)
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if pos.size == 0 or neg.size == 0:
            return float("nan")
        # Probability that a random positive is ranked higher than a random negative
        return float(np.mean(pos.reshape(-1, 1) > neg.reshape(1, -1)))


# ------------------------------
# Training
# ------------------------------
def _set_seeds(seed: int) -> None:
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            v.copy_(v * decay + msd[k] * (1.0 - decay))


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seeds(0 if args.seed_train is None else int(args.seed_train))

    # Datasets/loaders
    train_ds = PairIterableDataset(
        gen_name=args.generator,
        length=args.length,
        noise=args.noise,
        tau=args.tau,
        steps=args.train_steps * args.batch_size,
        pos_ratio=args.pos_ratio,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_margin=args.hard_margin,
        seed=args.seed_train,
    )
    valid_ds = PairIterableDataset(
        gen_name=args.generator,
        length=args.length,
        noise=args.noise,
        tau=args.tau,
        steps=args.valid_batches * args.batch_size,
        pos_ratio=args.pos_ratio,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_margin=args.hard_margin,
        seed=args.seed_valid,
    )
    test_ds = PairIterableDataset(
        gen_name=args.generator,
        length=args.length,
        noise=args.noise,
        tau=args.tau,
        steps=args.test_batches * args.batch_size,
        pos_ratio=args.pos_ratio,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_margin=args.hard_margin,
        seed=args.seed_test,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = CNNEncoder1D(in_channels=1, hidden=args.hidden, emb_dim=args.emb_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.train_steps // max(args.accum_steps, 1), 1))
    else:
        scheduler = None

    use_amp = bool(args.amp) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema_model: Optional[nn.Module] = None
    if args.ema_decay > 0.0:
        ema_model = deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    best_auc = -1.0
    steps = 0
    no_improve = 0

    # CSV logging
    csv_writer = None
    if args.log_csv:
        ensure_dir(args.log_csv)
        is_new = not os.path.exists(args.log_csv)
        csv_f = open(args.log_csv, "a", newline="")
        csv_writer = csv.writer(csv_f)
        if is_new:
            csv_writer.writerow(["step", "split", "loss", "auc", "acc", "mse", "lr"])  # header

    def run_eval(loader: DataLoader) -> Tuple[float, float, float]:
        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb1, xb2, yb in loader:
                # inputs come as (B, T, 1); model expects (B, 1, T)
                x1 = xb1.permute(0, 2, 1).to(device)
                x2 = xb2.permute(0, 2, 1).to(device)
                y = yb.to(device)
                z1 = eval_model(x1)
                z2 = eval_model(x2)
                cos = nn.functional.cosine_similarity(z1, z2)
                logits = args.scale * cos
                prob = torch.sigmoid(logits)
                ys.append(y.detach().cpu().numpy())
                ps.append(prob.detach().cpu().numpy())
        y_all = np.concatenate(ys, axis=0).astype(np.float64)
        p_all = np.concatenate(ps, axis=0).astype(np.float64)
        auc = auc_binary(y_all, p_all)
        acc = float(((p_all >= 0.5).astype(np.float32) == y_all).mean())
        mse = float(np.mean((y_all - p_all) ** 2))
        return mse, auc, acc

    accum = 0
    running_loss = 0.0
    for xb1, xb2, yb in train_loader:
        model.train()
        x1 = xb1.permute(0, 2, 1).to(device)
        x2 = xb2.permute(0, 2, 1).to(device)
        y = yb.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            z1 = model(x1)
            z2 = model(x2)
            cos = nn.functional.cosine_similarity(z1, z2)
            logits = args.scale * cos
            y_smooth = y
            if args.label_smoothing > 0:
                y_smooth = y * (1 - 2 * args.label_smoothing) + args.label_smoothing
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_smooth)
            loss = loss / max(args.accum_steps, 1)

        scaler.scale(loss).backward()
        accum += 1
        running_loss += loss.item()

        if accum >= max(args.accum_steps, 1):
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema_model is not None:
                _update_ema(ema_model, model, args.ema_decay)
            steps += 1
            accum = 0

            if steps % args.log_every == 0:
                with torch.no_grad():
                    prob = torch.sigmoid(logits)
                    acc = ((prob >= 0.5).float() == y).float().mean().item()
                lr = opt.param_groups[0]["lr"]
                print(f"step {steps:6d} | lr {lr:.5g} | loss {running_loss:.4f} | acc {acc:.3f}")
                if csv_writer:
                    csv_writer.writerow([steps, "train", f"{running_loss:.6f}", "", f"{acc:.6f}", "", f"{lr:.6g}"])
                running_loss = 0.0

            if steps % args.valid_every == 0:
                val_mse, val_auc, val_acc = run_eval(valid_loader)
                lr = opt.param_groups[0]["lr"]
                print(f"[valid] step {steps:6d} | lr {lr:.5g} | AUC {val_auc:.3f} | ACC {val_acc:.3f} | MSE {val_mse:.4f}")
                if csv_writer:
                    csv_writer.writerow([steps, "valid", "", f"{val_auc:.6f}", f"{val_acc:.6f}", f"{val_mse:.6f}", f"{lr:.6g}"])
                improved = (not np.isnan(val_auc)) and (val_auc > best_auc)
                if improved:
                    best_auc = val_auc
                    no_improve = 0
                    ensure_dir(args.ckpt)
                    to_save = ema_model if ema_model is not None else model
                    torch.save({
                        "encoder": to_save.state_dict(),
                        "args": {"hidden": args.hidden, "emb_dim": args.emb_dim},
                    }, args.ckpt)
                    print(f"Saved best checkpoint to {args.ckpt} (AUC={best_auc:.3f})")
                else:
                    no_improve += 1
                    if args.patience > 0 and no_improve >= args.patience:
                        print(f"Early stopping after {steps} steps without AUC improvement.")
                        break

            if steps >= args.train_steps:
                break

    # Final test on best model (if saved)
    if os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["encoder"])  # type: ignore
        print("Loaded best checkpoint for test evaluation.")
    test_mse, test_auc, test_acc = run_eval(test_loader)
    print(f"[test] AUC {test_auc:.3f} | ACC {test_acc:.3f} | MSE {test_mse:.4f}")
    if csv_writer:
        lr = opt.param_groups[0]["lr"]
        csv_writer.writerow([steps, "test", "", f"{test_auc:.6f}", f"{test_acc:.6f}", f"{test_mse:.6f}", f"{lr:.6g}"])


def get_parser():
    p = argparse.ArgumentParser(description="Train Siamese-like encoder with alpha-similarity labels")
    p.add_argument("--generator", choices=["sine", "wavelet"], default="wavelet")
    p.add_argument("--length", type=int, default=1000)
    p.add_argument("--noise", type=float, default=0.25)
    p.add_argument("--tau", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--valid-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--valid-batches", type=int, default=10)
    p.add_argument("--test-batches", type=int, default=20)
    p.add_argument("--pos-ratio", type=float, default=0.5)
    p.add_argument("--hard-neg-ratio", type=float, default=0.5)
    p.add_argument("--hard-margin", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    p.add_argument("--patience", type=int, default=5, help="Valid cycles without improvement before early stop (0=off)")
    p.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for weights (0=off)")
    p.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")
    p.add_argument("--seed-train", type=int, default=10)
    p.add_argument("--seed-valid", type=int, default=11)
    p.add_argument("--seed-test", type=int, default=12)
    p.add_argument("--ckpt", type=str, default=os.path.join("PULSOVITAL", "results", "similarity_model.pt"))
    p.add_argument("--log-csv", type=str, default=os.path.join("PULSOVITAL", "results", "similarity_log.csv"))
    return p


def main():
    args = get_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
