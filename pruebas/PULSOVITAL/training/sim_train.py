import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader

try:
    from PULSOVITAL.core.fid_all_in_one import SineDataGenerator, WaveletDataGenerator, ensure_dir
except Exception:
    # Fallback to root shim if present
    from PULSOVITAL.fid_all_in_one import SineDataGenerator, WaveletDataGenerator, ensure_dir


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
    ):
        gens = {"sine": SineDataGenerator, "wavelet": WaveletDataGenerator}
        if gen_name not in gens:
            raise ValueError(f"Unknown generator: {gen_name}")
        self.Gen = gens[gen_name]
        self.length = length
        self.noise = noise
        self.tau = float(tau)
        self.steps = int(steps)
        self.pos_ratio = float(pos_ratio)
        self.hard_neg_ratio = float(hard_neg_ratio)
        self.hard_margin = float(hard_margin)
        self.seed = int(seed)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + (torch.initial_seed() % (2**32 - 1)))
        G = self.Gen(length=self.length, noise_scale=self.noise)
        i = 0
        while i < self.steps:
            a1 = rng.uniform(0.0, 1.0)
            is_pos = rng.random() < self.pos_ratio
            if is_pos:
                a2 = np.clip(a1 + rng.uniform(-self.tau, self.tau), 0.0, 1.0)
                y = 1.0
            else:
                if rng.random() < self.hard_neg_ratio:
                    shift = self.tau + rng.uniform(0.0, self.hard_margin)
                    a2 = np.clip(a1 + (shift if rng.random() < 0.5 else -shift), 0.0, 1.0)
                else:
                    a2 = rng.uniform(0.0, 1.0)
                    if abs(a2 - a1) <= self.tau:
                        a2 = np.clip(a1 + np.sign(a2 - a1) * (self.tau + self.hard_margin), 0.0, 1.0)
                y = 0.0
            x1 = G.sample(a1)
            x2 = G.sample(a2)
            i += 1
            yield x1.float(), x2.float(), torch.tensor(y, dtype=torch.float32)


class CNNEncoder1D(nn.Module):
    def __init__(self, in_channels: int = 1, hidden: int = 64, emb_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, hidden, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(hidden, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).mean(dim=-1)
        z = self.proj(h)
        z = nn.functional.normalize(z, dim=1)
        return z


def auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y_score) + 1, dtype=float)
    rank_pos = ranks[y_true == 1].sum()
    U = rank_pos - pos * (pos + 1) / 2.0
    return float(U / (pos * neg))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PairIterableDataset(
        gen_name=args.generator,
        length=args.length,
        noise=args.noise,
        tau=args.tau,
        steps=args.train_steps,
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

    best_auc = -1.0
    steps = 0

    def run_eval(loader: DataLoader) -> Tuple[float, float, float]:
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb1, xb2, yb in loader:
                x1 = xb1.permute(0, 2, 1).to(device)
                x2 = xb2.permute(0, 2, 1).to(device)
                y = yb.to(device)
                z1 = model(x1)
                z2 = model(x2)
                cos = nn.functional.cosine_similarity(z1, z2)
                logits = args.scale * cos
                prob = torch.sigmoid(logits)
                ys.append(y.detach().cpu().numpy())
                ps.append(prob.detach().cpu().numpy())
        y_all = np.concatenate(ys, axis=0).astype(np.float64)
        p_all = np.concatenate(ps, axis=0).astype(np.float64)
        auc = auc_binary(y_all, p_all)
        acc = float(((p_all >= 0.5).astype(np.float32) == y_all).mean())
        return float(np.mean((y_all - p_all) ** 2)), auc, acc

    for xb1, xb2, yb in train_loader:
        model.train()
        x1 = xb1.permute(0, 2, 1).to(device)
        x2 = xb2.permute(0, 2, 1).to(device)
        y = yb.to(device)

        z1 = model(x1)
        z2 = model(x2)
        cos = nn.functional.cosine_similarity(z1, z2)
        logits = args.scale * cos

        if args.label_smoothing > 0:
            y = y * (1 - 2 * args.label_smoothing) + args.label_smoothing
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        steps += 1
        if steps % args.log_every == 0:
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                acc = ((prob >= 0.5).float() == y).float().mean().item()
            print(f"step {steps:6d} | loss {loss.item():.4f} | acc {acc:.3f}")

        if steps % args.valid_every == 0:
            val_mse, val_auc, val_acc = run_eval(valid_loader)
            print(f"[valid] step {steps:6d} | AUC {val_auc:.3f} | ACC {val_acc:.3f} | MSE {val_mse:.4f}")
            if val_auc > best_auc:
                best_auc = val_auc
                ensure_dir(args.ckpt)
                torch.save({
                    "encoder": model.state_dict(),
                    "args": {"hidden": args.hidden, "emb_dim": args.emb_dim}
                }, args.ckpt)
                print(f"Saved best checkpoint to {args.ckpt} (AUC={best_auc:.3f})")

        if steps >= args.train_steps:
            break

    # Final test on best model (if saved)
    if os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["encoder"])  # type: ignore
        print("Loaded best checkpoint for test evaluation.")
    test_mse, test_auc, test_acc = run_eval(test_loader)
    print(f"[test] AUC {test_auc:.3f} | ACC {test_acc:.3f} | MSE {test_mse:.4f}")


def get_parser():
    p = argparse.ArgumentParser(description="Train Siamese-like encoder with alpha-similarity labels")
    p.add_argument("--generator", choices=["sine", "wavelet"], default="wavelet")
    p.add_argument("--length", type=int, default=1000)
    p.add_argument("--noise", type=float, default=0.25)
    p.add_argument("--tau", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=1200)
    p.add_argument("--valid-every", type=int, default=400)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--valid-batches", type=int, default=8)
    p.add_argument("--test-batches", type=int, default=12)
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
    p.add_argument("--seed-train", type=int, default=10)
    p.add_argument("--seed-valid", type=int, default=11)
    p.add_argument("--seed-test", type=int, default=12)
    p.add_argument("--ckpt", type=str, default=os.path.join("PULSOVITAL", "results", "similarity_model.pt"))
    return p


def main():
    args = get_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
