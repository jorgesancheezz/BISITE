import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Reuse data pipeline and FID utilities from core to avoid duplication
try:
    from PULSOVITAL.core.fid_all_in_one import (
        build_loader,
        parse_kwargs,
        compute_stats,
        frange,
        ensure_dir,
    )
except Exception:
    import sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(__file__))
    if _root not in sys.path:
        sys.path.append(_root)
    from PULSOVITAL.core.fid_all_in_one import (
        build_loader,
        parse_kwargs,
        compute_stats,
        frange,
        ensure_dir,
    )

# Local copy of Frechet distance with SciPy sqrtm compatibility kept inside this file
from scipy import linalg

def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
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


##############################################
# Encoder model (same as in similarity_train)
##############################################
class CNNEncoder1D(nn.Module):
    def __init__(self, in_channels: int = 1, hidden: int = 64, emb_dim: int = 64):
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
        # x: (B, C, T)
        h = self.net(x)
        h = h.mean(dim=-1)  # global average pool over time
        z = self.proj(h)
        z = nn.functional.normalize(z, dim=1)
        return z


def load_encoder(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, int, int]:
    """Load trained encoder from checkpoint and return (model, hidden, emb_dim)."""
    state = torch.load(ckpt_path, map_location=device)
    enc_args = state.get("args", {}) or {}
    hidden = int(enc_args.get("hidden", 64))
    emb_dim = int(enc_args.get("emb_dim", 64))
    model = CNNEncoder1D(in_channels=1, hidden=hidden, emb_dim=emb_dim).to(device)
    try:
        model.load_state_dict(state["encoder"], strict=False)
    except Exception as e:
        print(f"Warning: partial/failed load of encoder weights: {e}")
    model.eval()
    return model, hidden, emb_dim


@torch.no_grad()
def collect_embeddings(encoder: nn.Module, loader: DataLoader, max_samples: int, device: torch.device) -> np.ndarray:
    """Collect normalized embeddings from encoder for up to max_samples items.
    Input batches come as (B, T, 1); encoder expects (B, 1, T).
    """
    embs = []
    collected = 0
    for xb in loader:
        xb = xb.to(dtype=torch.float32, device=device)  # (B, T, 1)
        x = xb.permute(0, 2, 1)                         # (B, 1, T)
        z = encoder(x)                                  # (B, D)
        embs.append(z.detach().cpu().numpy().astype(np.float32))
        collected += z.shape[0]
        if collected >= max_samples:
            break
    if not embs:
        return np.empty((0, 0), dtype=np.float32)
    all_embs = np.concatenate(embs, axis=0)
    return all_embs[:max_samples]


##############################
# CLI
##############################

def run_pair(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _, _ = load_encoder(args.ckpt, device)
    ref_kwargs = parse_kwargs(args.ref_generator_kwargs)
    cmp_kwargs = parse_kwargs(args.cmp_generator_kwargs)

    ref_loader = build_loader(
        args.ref_generator,
        args.length,
        args.ref_alpha,
        args.ref_noise,
        args.batch_size,
        args.samples,
        args.num_workers,
        ref_kwargs,
    )
    cmp_loader = build_loader(
        args.cmp_generator,
        args.length,
        args.cmp_alpha,
        args.cmp_noise,
        args.batch_size,
        args.samples,
        args.num_workers,
        cmp_kwargs,
    )

    ref_embs = collect_embeddings(encoder, ref_loader, args.samples, device)
    cmp_embs = collect_embeddings(encoder, cmp_loader, args.samples, device)

    ref_embs = ref_embs[np.isfinite(ref_embs).all(axis=1)]
    cmp_embs = cmp_embs[np.isfinite(cmp_embs).all(axis=1)]
    if ref_embs.size == 0 or cmp_embs.size == 0:
        raise ValueError("No valid embeddings collected to compute FID.")

    mu1, sigma1 = compute_stats(ref_embs)
    mu2, sigma2 = compute_stats(cmp_embs)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"Model-FID: {fid:.6f}")


def run_sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _, _ = load_encoder(args.ckpt, device)
    ref_kwargs = parse_kwargs(args.ref_generator_kwargs)
    cmp_kwargs = parse_kwargs(args.cmp_generator_kwargs)

    ref_loader = build_loader(
        args.ref_generator,
        args.length,
        args.ref_alpha,
        args.ref_noise,
        args.batch_size,
        args.samples,
        args.num_workers,
        ref_kwargs,
    )
    ref_embs = collect_embeddings(encoder, ref_loader, args.samples, device)
    ref_embs = ref_embs[np.isfinite(ref_embs).all(axis=1)]
    if ref_embs.size == 0:
        raise ValueError("No valid reference embeddings for sweep.")
    mu_ref, sigma_ref = compute_stats(ref_embs)

    alphas = frange(args.alpha_start, args.alpha_stop, args.alpha_step)
    fids = []
    for a in alphas:
        cmp_loader = build_loader(
            args.cmp_generator,
            args.length,
            a,
            args.cmp_noise,
            args.batch_size,
            args.samples,
            args.num_workers,
            cmp_kwargs,
        )
        cmp_embs = collect_embeddings(encoder, cmp_loader, args.samples, device)
        cmp_embs = cmp_embs[np.isfinite(cmp_embs).all(axis=1)]
        if cmp_embs.size == 0:
            raise ValueError(f"No valid comparison embeddings for alpha={a}.")
        mu_cmp, sigma_cmp = compute_stats(cmp_embs)
        fid = frechet_distance(mu_ref, sigma_ref, mu_cmp, sigma_cmp)
        fids.append(fid)
        print(f"alpha={a:.3f} -> Model-FID={fid:.6f}")

    # CSV (wide, append new FID column)
    ensure_dir(args.out_csv)
    import csv
    fid_series = {float(a): float(v) for a, v in zip(alphas, fids)}

    def fmt_alpha(a: float) -> str:
        s = f"{a:.10f}"
        s = s.rstrip("0").rstrip(".") if "." in s else s
        return s

    if os.path.exists(args.out_csv):
        with open(args.out_csv, "r", newline="") as f:
            r = csv.reader(f)
            rows = list(r)
        header = rows[0] if rows else ["alpha"]
        data_rows = rows[1:] if rows else []
        fid_cols = [c for c in header if c.lower().startswith("fid")]
        if "FID" in header and "FID 1" not in header:
            idx = header.index("FID")
            header[idx] = "FID 1"
        # determine next index robustly from numeric suffixes
        fid_cols = [c for c in header if c.lower().startswith("fid")]
        suffixes = []
        for c in fid_cols:
            parts = c.split()
            if len(parts) == 2 and parts[0].upper() == "FID":
                try:
                    suffixes.append(int(parts[1]))
                except Exception:
                    pass
            elif c.upper() == "FID":
                suffixes.append(1)
        next_num = (max(suffixes) + 1) if suffixes else 1
        new_col = f"FID {next_num}"
        # build existing map
        existing = {}
        for r in data_rows:
            if not r:
                continue
            try:
                a = float(r[0])
            except Exception:
                try:
                    a = float(r[0].replace(",", "."))
                except Exception:
                    continue
            row_dict = {header[i]: r[i] if i < len(r) else "" for i in range(len(header))}
            existing[a] = row_dict
        all_alphas = sorted(set(existing.keys()).union(set(fid_series.keys())))
        if header and header[0].lower() != "alpha":
            header = ["alpha"] + [h for h in header if h.lower() != "alpha"]
        out_header = ["alpha"] + [c for c in header[1:] if c.lower().startswith("fid")] + [new_col]
        out_rows = []
        for a in all_alphas:
            row = {h: "" for h in out_header}
            row["alpha"] = fmt_alpha(a)
            if a in existing:
                for h in out_header:
                    if h in existing[a]:
                        row[h] = existing[a][h]
            if a in fid_series:
                row[new_col] = f"{fid_series[a]:.6f}"
            out_rows.append([row[h] for h in out_header])
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(out_header)
            w.writerows(out_rows)
    else:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["alpha", "FID 1"])
            for a in alphas:
                w.writerow([fmt_alpha(float(a)), f"{fid_series[float(a)]:.6f}"])
    print(f"Saved CSV to {args.out_csv}")


################
# Utils
################

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def get_parser():
    _generators = ("sine", "wavelet")
    parser = argparse.ArgumentParser(description="FID using trained model embeddings")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Pair mode
    p = sub.add_parser("pair", help="Compute a single FID between REF and CMP using model embeddings")
    p.add_argument("--ckpt", type=str, default=os.path.join("results", "similarity_model.pt"))
    p.add_argument("--length", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--ref-generator", choices=_generators, default="sine")
    p.add_argument("--ref-alpha", type=float, required=True)
    p.add_argument("--ref-noise", type=float, default=0.0)
    p.add_argument("--ref-generator-kwargs", nargs="+", default=[])
    p.add_argument("--cmp-generator", choices=_generators, default="wavelet")
    p.add_argument("--cmp-alpha", type=float, required=True)
    p.add_argument("--cmp-noise", type=float, default=0.0)
    p.add_argument("--cmp-generator-kwargs", nargs="+", default=[])

    # Sweep mode
    s = sub.add_parser("sweep", help="Sweep CMP alpha and save CSV (+ append column) using model embeddings")
    s.add_argument("--ckpt", type=str, default=os.path.join("results", "similarity_model.pt"))
    s.add_argument("--length", type=int, default=1000)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--num-workers", type=int, default=4)
    s.add_argument("--samples", type=int, default=1024)
    s.add_argument("--ref-generator", choices=_generators, default="sine")
    s.add_argument("--ref-alpha", type=float, required=True)
    s.add_argument("--ref-noise", type=float, default=0.0)
    s.add_argument("--ref-generator-kwargs", nargs="+", default=[])
    s.add_argument("--cmp-generator", choices=_generators, default="wavelet")
    s.add_argument("--cmp-noise", type=float, default=0.0)
    s.add_argument("--cmp-generator-kwargs", nargs="+", default=[])
    s.add_argument("--alpha-start", type=float, default=0.0)
    s.add_argument("--alpha-stop", type=float, default=1.0)
    s.add_argument("--alpha-step", type=float, default=0.1)
    s.add_argument("--out-csv", type=str, default=os.path.join("results", "fid_sweep_model.csv"))

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.mode == "pair":
        run_pair(args)
    elif args.mode == "sweep":
        run_sweep(args)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
