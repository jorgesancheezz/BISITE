import argparse
import os
import csv
from typing import List

import numpy as np
import torch

# Reuse utilities from core and root shim for model helpers
try:
    from PULSOVITAL.core.fid_all_in_one import (
        build_loader,
        parse_kwargs,
        compute_stats,
        frange,
        collect_features,
        ensure_dir,
        frechet_distance,
    )
    from PULSOVITAL.metrics.fid_with_model import (
        load_encoder,
        collect_embeddings,
    )
except Exception:
    # Fallback for direct execution without package context
    import sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(__file__))
    if _root not in sys.path:
        sys.path.append(_root)
    from PULSOVITAL.core.fid_all_in_one import (
        build_loader,
        parse_kwargs,
        compute_stats,
        frange,
        collect_features,
        ensure_dir,
        frechet_distance,
    )
    from PULSOVITAL.metrics.fid_with_model import (
        load_encoder,
        collect_embeddings,
    )


def run_sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_kwargs = parse_kwargs(args.ref_generator_kwargs)
    cmp_kwargs = parse_kwargs(args.cmp_generator_kwargs)

    # Load model encoder
    encoder, _, _ = load_encoder(args.ckpt, device)

    # Reference loader and features
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
    ref_spec = collect_features(ref_loader, args.samples, device)
    ref_spec = ref_spec[np.isfinite(ref_spec).all(axis=1)]
    if ref_spec.size == 0:
        raise ValueError("No valid reference spectrogram features.")
    mu_ref_spec, sig_ref_spec = compute_stats(ref_spec)

    # Need a fresh ref loader for embeddings to avoid exhausting iterator
    ref_loader_emb = build_loader(
        args.ref_generator,
        args.length,
        args.ref_alpha,
        args.ref_noise,
        args.batch_size,
        args.samples,
        args.num_workers,
        ref_kwargs,
    )
    ref_emb = collect_embeddings(encoder, ref_loader_emb, args.samples, device)
    ref_emb = ref_emb[np.isfinite(ref_emb).all(axis=1)]
    if ref_emb.size == 0:
        raise ValueError("No valid reference embeddings.")
    mu_ref_emb, sig_ref_emb = compute_stats(ref_emb)

    alphas = frange(args.alpha_start, args.alpha_stop, args.alpha_step)
    fids_spec: List[float] = []
    fids_model: List[float] = []

    for a in alphas:
        # Comparison spectrogram features
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
        cmp_spec = collect_features(cmp_loader, args.samples, device)
        cmp_spec = cmp_spec[np.isfinite(cmp_spec).all(axis=1)]
        if cmp_spec.size == 0:
            raise ValueError(f"No valid comparison spectrogram features for alpha={a}.")
        mu_cmp_spec, sig_cmp_spec = compute_stats(cmp_spec)
        fid_spec = frechet_distance(mu_ref_spec, sig_ref_spec, mu_cmp_spec, sig_cmp_spec)

        # Comparison embeddings (fresh loader)
        cmp_loader_emb = build_loader(
            args.cmp_generator,
            args.length,
            a,
            args.cmp_noise,
            args.batch_size,
            args.samples,
            args.num_workers,
            cmp_kwargs,
        )
        cmp_emb = collect_embeddings(encoder, cmp_loader_emb, args.samples, device)
        cmp_emb = cmp_emb[np.isfinite(cmp_emb).all(axis=1)]
        if cmp_emb.size == 0:
            raise ValueError(f"No valid comparison embeddings for alpha={a}.")
        mu_cmp_emb, sig_cmp_emb = compute_stats(cmp_emb)
        fid_model = frechet_distance(mu_ref_emb, sig_ref_emb, mu_cmp_emb, sig_cmp_emb)

        fids_spec.append(float(fid_spec))
        fids_model.append(float(fid_model))
        print(f"alpha={a:.3f} -> FID_spec={fid_spec:.6f} | FID_model={fid_model:.6f}")

    # Write combined CSV (do not append; create new file each run unless --append)
    ensure_dir(args.out_csv)
    if args.append and os.path.exists(args.out_csv):
        # Append two new columns numbered based on existing header
        with open(args.out_csv, "r", newline="") as f:
            r = csv.reader(f)
            rows = list(r)
        header = rows[0] if rows else ["alpha"]
        data_rows = rows[1:] if rows else []

        # Determine next pair index
        existing_fid_cols = [h for h in header if h.lower().startswith("fid spec") or h.lower().startswith("fid model")]
        # Find the largest suffix like "FID spec N"
        def get_idx(h):
            parts = h.strip().split()
            try:
                return int(parts[-1]) if parts[-1].isdigit() else None
            except Exception:
                return None
        suffixes = [i for h in existing_fid_cols for i in ([get_idx(h)] if get_idx(h) is not None else [])]
        next_idx = (max(suffixes) + 1) if suffixes else 1
        col_spec = f"FID spec {next_idx}"
        col_model = f"FID model {next_idx}"

        # Build alpha->row map
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
        all_alphas = sorted(set(existing.keys()).union(set(map(float, alphas))))

        out_header = ["alpha"] + [h for h in header[1:]] + [col_spec, col_model]
        out_rows = []
        series_spec = {float(a): float(v) for a, v in zip(alphas, fids_spec)}
        series_model = {float(a): float(v) for a, v in zip(alphas, fids_model)}
        for a in all_alphas:
            row = {h: "" for h in out_header}
            row["alpha"] = f"{a:.10f}".rstrip("0").rstrip(".")
            if a in existing:
                for h in header[1:]:
                    if h in row:
                        row[h] = existing[a].get(h, "")
            if a in series_spec:
                row[col_spec] = f"{series_spec[a]:.6f}"
            if a in series_model:
                row[col_model] = f"{series_model[a]:.6f}"
            out_rows.append([row[h] for h in out_header])
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(out_header)
            w.writerows(out_rows)
    else:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["alpha", "FID spec", "FID model"])
            for a, v_spec, v_model in zip(alphas, fids_spec, fids_model):
                s = f"{float(a):.10f}".rstrip("0").rstrip(".")
                w.writerow([s, f"{v_spec:.6f}", f"{v_model:.6f}"])
    print(f"Saved CSV to {args.out_csv}")


def get_parser():
    _generators = ("sine", "wavelet")
    p = argparse.ArgumentParser(description="Compare FID using spectrogram features vs model embeddings")
    p.add_argument("--ckpt", type=str, default=os.path.join("PULSOVITAL", "results", "similarity_model.pt"))
    p.add_argument("--length", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--ref-generator", choices=_generators, default="sine")
    p.add_argument("--ref-alpha", type=float, required=True)
    p.add_argument("--ref-noise", type=float, default=0.0)
    p.add_argument("--ref-generator-kwargs", nargs="+", default=[])
    p.add_argument("--cmp-generator", choices=_generators, default="wavelet")
    p.add_argument("--cmp-noise", type=float, default=0.0)
    p.add_argument("--cmp-generator-kwargs", nargs="+", default=[])
    p.add_argument("--alpha-start", type=float, default=0.0)
    p.add_argument("--alpha-stop", type=float, default=1.0)
    p.add_argument("--alpha-step", type=float, default=0.1)
    p.add_argument("--out-csv", type=str, default=os.path.join("PULSOVITAL", "results", "fid_compare_both.csv"))
    p.add_argument("--append", action="store_true", help="Append as new paired columns if out CSV exists")
    return p


def main():
    args = get_parser().parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
