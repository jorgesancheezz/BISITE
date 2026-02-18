import argparse
import csv
import os
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_compare_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
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

    # Detect columns for spec/model (base + con sufijos)
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
    A = np.array(alphas, dtype=np.float64)
    S = np.array([row for row in spec_vals], dtype=np.float64)
    M = np.array([row for row in model_vals], dtype=np.float64)
    return A, S, M, header


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def plot_mean_std(alphas: np.ndarray, values: np.ndarray, *, label: str, color: str, ax: plt.Axes, yscale: str = "linear"):
    mu = np.nanmean(values, axis=1)
    sd = np.nanstd(values, axis=1)
    if yscale == "log":
        # evitar ceros
        mu = np.clip(mu, 1e-12, None)
        sd = np.clip(sd, 1e-12, None)
    ax.plot(alphas, mu, marker="o", color=color, label=label)
    ax.fill_between(alphas, mu - sd, mu + sd, color=color, alpha=0.2)


def main():
    ap = argparse.ArgumentParser(description="Plot mean±std de FID spec vs FID model en función de alpha")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out-plot", type=str, default=None)
    ap.add_argument("--yscale", choices=["linear", "log"], default="linear")
    args = ap.parse_args()

    A, S, M, _ = read_compare_csv(args.csv)
    order = np.argsort(A)
    A = A[order]
    S = S[order]
    M = M[order]

    if args.out_plot is None:
        base, _ = os.path.splitext(args.csv)
        out = base + "_mean_std.png"
    else:
        out = args.out_plot
    ensure_dir(out)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    if args.yscale == "log":
        ax.set_yscale("log")

    plot_mean_std(A, S, label="FID spec (μ±σ)", color="#1f77b4", ax=ax, yscale=args.yscale)
    plot_mean_std(A, M, label="FID model (μ±σ)", color="#d62728", ax=ax, yscale=args.yscale)

    ax.set_xlabel("alpha")
    ax.set_ylabel("FID")
    ax.set_title("FID spec vs model (media ± desviación)")
    ax.grid(True, alpha=0.3)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_extra_artists=(leg,), bbox_inches="tight")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
