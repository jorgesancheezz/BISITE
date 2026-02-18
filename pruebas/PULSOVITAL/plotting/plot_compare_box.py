import argparse
import csv
import os
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def read_compare_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        raise ValueError("CSV vacío")
    header = rows[0]
    data = rows[1:]
    try:
        aidx = header.index("alpha")
    except ValueError:
        raise ValueError("CSV debe tener columna 'alpha'")

    spec_cols = [i for i, h in enumerate(header) if h.lower().startswith("fid spec")]
    model_cols = [i for i, h in enumerate(header) if h.lower().startswith("fid model")]
    if not spec_cols or not model_cols:
        raise ValueError("No se encontraron columnas 'FID spec*' y 'FID model*'")

    alphas: List[float] = []
    spec_vals: List[List[float]] = []
    model_vals: List[List[float]] = []
    for r in data:
        if not r:
            continue
        try:
            a = float(r[aidx])
        except Exception:
            try:
                a = float(r[aidx].replace(",", "."))
            except Exception:
                continue
        srow = []
        mrow = []
        for i in spec_cols:
            if i < len(r) and r[i] != "":
                try:
                    srow.append(float(r[i]))
                except Exception:
                    pass
        for i in model_cols:
            if i < len(r) and r[i] != "":
                try:
                    mrow.append(float(r[i]))
                except Exception:
                    pass
        if srow and mrow:
            alphas.append(a)
            spec_vals.append(srow)
            model_vals.append(mrow)

    if not alphas:
        raise ValueError("No hay filas válidas en el CSV")
    return np.array(alphas, dtype=np.float64), np.array(spec_vals, dtype=np.float64), np.array(model_vals, dtype=np.float64)


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Box plots agrupados por alpha: FID spec vs FID model")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out-plot", type=str, default=None)
    ap.add_argument("--yscale", choices=["linear", "log"], default="linear")
    ap.add_argument("--alpha-min", type=float, default=None)
    ap.add_argument("--alpha-max", type=float, default=None)
    args = ap.parse_args()

    A, S, M = read_compare_csv(args.csv)

    # filter by alpha if requested
    mask = np.ones_like(A, dtype=bool)
    if args.alpha_min is not None:
        mask &= (A >= args.alpha_min)
    if args.alpha_max is not None:
        mask &= (A <= args.alpha_max)
    A, S, M = A[mask], S[mask], M[mask]
    if A.size == 0:
        raise ValueError("Filtrado de alpha eliminó todas las filas")

    # sort by alpha
    order = np.argsort(A)
    A, S, M = A[order], S[order], M[order]

    if args.out_plot is None:
        base, _ = os.path.splitext(args.csv)
        suffix = "_box.png" if args.yscale == "linear" else "_box_log.png"
        out = base + suffix
    else:
        out = args.out_plot
    ensure_dir(out)

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    if args.yscale == "log":
        ax.set_yscale("log")

    n = len(A)
    x = np.arange(n)
    width = 0.35
    pos_spec = x - width/2
    pos_model = x + width/2

    # Build data lists per alpha; clip zeros if log
    data_spec = []
    data_model = []
    for i in range(n):
        sv = S[i, :]
        mv = M[i, :]
        sv = sv[np.isfinite(sv)]
        mv = mv[np.isfinite(mv)]
        if args.yscale == "log":
            sv = np.clip(sv, 1e-12, None)
            mv = np.clip(mv, 1e-12, None)
        data_spec.append(sv.tolist())
        data_model.append(mv.tolist())

    # Plot boxplots
    bp_spec = ax.boxplot(
        data_spec,
        positions=pos_spec,
        widths=width*0.9,
        patch_artist=True,
        showmeans=False,
        boxprops=dict(facecolor="#1f77b4", color="#1f77b4", alpha=0.5),
        medianprops=dict(color="#1f77b4"),
        whiskerprops=dict(color="#1f77b4"),
        capprops=dict(color="#1f77b4"),
        flierprops=dict(markeredgecolor="#1f77b4", marker='o', markersize=2, alpha=0.5),
    )
    bp_model = ax.boxplot(
        data_model,
        positions=pos_model,
        widths=width*0.9,
        patch_artist=True,
        showmeans=False,
        boxprops=dict(facecolor="#d62728", color="#d62728", alpha=0.5),
        medianprops=dict(color="#d62728"),
        whiskerprops=dict(color="#d62728"),
        capprops=dict(color="#d62728"),
        flierprops=dict(markeredgecolor="#d62728", marker='o', markersize=2, alpha=0.5),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.1f}".rstrip('0').rstrip('.') for a in A])
    ax.set_xlabel("alpha")
    ax.set_ylabel("FID")
    ax.set_title("Box plot por alpha: FID spec vs FID model")
    ax.grid(True, axis="y", alpha=0.3)

    # Legend proxies
    legend_patches = [
        Patch(facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.5, label="FID spec"),
        Patch(facecolor="#d62728", edgecolor="#d62728", alpha=0.5, label="FID model"),
    ]
    leg = ax.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_extra_artists=(leg,), bbox_inches="tight")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
