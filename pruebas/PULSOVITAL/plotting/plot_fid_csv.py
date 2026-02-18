import argparse
import csv
import os
import re
import glob
from typing import List, Tuple, Optional

# Force non-interactive backend for reliability
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.express as px
    _PLOTLY_OK = True
except Exception:
    px = None
    _PLOTLY_OK = False


def read_fid_csv(csv_path: str) -> Tuple[List[float], dict]:
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        raise ValueError("CSV is empty")
    header = rows[0]
    data_rows = rows[1:]
    # Identify columns
    try:
        alpha_idx = header.index("alpha")
    except ValueError:
        raise ValueError("CSV must have an 'alpha' column")
    fid_cols = [(i, h) for i, h in enumerate(header) if h.lower().startswith("fid")]
    # Parse
    alphas = []
    series = {name: [] for _, name in fid_cols}
    for r in data_rows:
        if not r:
            continue
        try:
            a = float(r[alpha_idx])
        except Exception:
            continue
        alphas.append(a)
        for i, name in fid_cols:
            val = None
            if i < len(r) and r[i] != "":
                try:
                    val = float(r[i])
                except Exception:
                    val = None
            series[name].append(val)
    return alphas, series


def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def find_latest_csv(search_dir: str = os.path.join("results")) -> Optional[str]:
    if not os.path.isdir(search_dir):
        return None
    candidates = glob.glob(os.path.join(search_dir, "*.csv"))
    if not candidates:
        return None
    # Prefer names containing 'fid' or 'sweep'
    def score(p: str) -> tuple:
        name = os.path.basename(p).lower()
        pref = 0
        if "fid" in name:
            pref -= 2
        if "sweep" in name:
            pref -= 1
        return (pref, -os.path.getmtime(p))
    best = sorted(candidates, key=score)[0]
    return best


def main():
    ap = argparse.ArgumentParser(description="Plot FID CSV; auto-detect latest CSV in results/ if --csv not provided")
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV. If omitted, picks most recent CSV under results/ (prefers names with 'fid'/'sweep')")
    ap.add_argument("--out-plot", type=str, default=None, help="Output PNG. Defaults to <csv_basename>_plot.png next to the CSV")
    ap.add_argument("--out-html", type=str, default=None, help="Output HTML. Defaults to <csv_basename>_plot.html next to the CSV")
    ap.add_argument("--x-axis", choices=["alpha", "fid"], default="fid", help="alpha: x=alpha, lines=FID series; fid: x=FID series, lines=alpha")
    ap.add_argument("--yscale", choices=["linear", "log"], default="linear", help="Y axis scale (use 'log' to resaltar diferencias pequeñas)")
    ap.add_argument("--alpha-min", type=float, default=None, help="Optional minimum alpha to include (inclusive)")
    ap.add_argument("--alpha-max", type=float, default=None, help="Optional maximum alpha to include (inclusive)")
    args = ap.parse_args()

    # Resolve CSV path (auto-detect if needed)
    csv_path = args.csv
    if csv_path is None or not os.path.isfile(csv_path):
        autodetected = find_latest_csv(os.path.join("results"))
        if autodetected is None:
            raise FileNotFoundError("No CSV provided and none found under 'results/'. Pass --csv path explicitly.")
        csv_path = autodetected
        print(f"Auto-detected CSV: {csv_path}")

    alphas, series = read_fid_csv(csv_path)

    # Optional alpha filtering
    if args.alpha_min is not None or args.alpha_max is not None:
        mask = []
        for a in alphas:
            ok = True
            if args.alpha_min is not None:
                ok = ok and (a >= args.alpha_min)
            if args.alpha_max is not None:
                ok = ok and (a <= args.alpha_max)
            mask.append(ok)
        # Apply mask
        if not any(mask):
            raise ValueError("Alpha filter excluded all rows; adjust --alpha-min/--alpha-max")
        alphas = [a for a, m in zip(alphas, mask) if m]
        for name, values in list(series.items()):
            series[name] = [v for v, m in zip(values, mask) if m]

    # Matplotlib PNG
    # Derive default outputs if not provided
    if args.out_plot is None:
        base, _ = os.path.splitext(csv_path)
        out_plot = base + "_plot.png"
    else:
        out_plot = args.out_plot
    if args.out_html is None:
        base, _ = os.path.splitext(csv_path)
        out_html = base + "_plot.html"
    else:
        out_html = args.out_html

    ensure_dir(out_plot)
    plt.figure(figsize=(10.0, 4.8))

    if args.x_axis == "alpha":
        # x = alpha, separate lines per FID series (original behavior)
        for name, values in series.items():
            if not values:
                continue
            y = [np.nan if v is None else v for v in values]
            if args.yscale == "log":
                y = [np.nan if np.isnan(v) else max(v, 1e-12) for v in y]
            plt.plot(alphas, y, marker="o", label=name)
        plt.xlabel("alpha")
        plt.ylabel("FID")
        plt.title("FID vs alpha (lines = FID series)")
    else:
        # x = FID series, separate lines per alpha (requested behavior)
        # Sort FID names by their numeric suffix if present
        def fid_key(name: str):
            m = re.search(r"(\d+)$", name.replace(" ", ""))
            return (0, int(m.group(1))) if m else (1, name)

        fid_names = sorted(series.keys(), key=fid_key)
        # Determine available rows across all series
        n_rows = min(len(series[name]) for name in fid_names) if fid_names else 0
        if n_rows == 0:
            raise ValueError("No data rows found in CSV for FID columns")
        # Align alpha list to n_rows (in case of ragged data)
        alphas = alphas[:n_rows]
        # Build x positions for categorical axis
        x_pos = np.arange(len(fid_names))
        for row_idx, a in enumerate(alphas):
            y = []
            for name in fid_names:
                vlist = series.get(name, [])
                v = vlist[row_idx] if row_idx < len(vlist) else None
                val = np.nan if v is None else v
                if args.yscale == "log" and not (isinstance(val, float) and np.isnan(val)):
                    val = max(val, 1e-12)
                y.append(val)
            plt.plot(x_pos, y, marker="o", label=f"alpha={a}")
        plt.xticks(x_pos, fid_names, rotation=35, ha="right")
        plt.xlabel("FID series")
        plt.ylabel("FID value")
        plt.title("FID across series (lines = alpha)")

    # Apply Y scale
    try:
        plt.yscale(args.yscale)
    except Exception:
        pass

    plt.grid(True, alpha=0.3)
    # Legend: outside on the right, smaller font
    leg = plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=True)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150, bbox_extra_artists=(leg,), bbox_inches="tight")
    print(f"Saved plot to {out_plot}")

    # Optional Plotly HTML
    if _PLOTLY_OK:
        try:
            import pandas as pd
        except Exception:
            pd = None
        # Build a long-form dataframe
        data = []
        if args.x_axis == "alpha":
            for name, values in series.items():
                for a, v in zip(alphas, values):
                    if v is not None:
                        data.append({"alpha": a, "FID": v, "Series": name})
            x_col, y_col, color_col = "alpha", "FID", "Series"
        else:
            # Long form with x=FID series, y=FID value, color=alpha
            # Determine n_rows and ordered fid_names as above
            def fid_key(name: str):
                m = re.search(r"(\d+)$", name.replace(" ", ""))
                return (0, int(m.group(1))) if m else (1, name)
            fid_names = sorted(series.keys(), key=fid_key)
            n_rows = min(len(series[name]) for name in fid_names) if fid_names else 0
            alphas_trim = alphas[:n_rows]
            for row_idx, a in enumerate(alphas_trim):
                for name in fid_names:
                    vlist = series.get(name, [])
                    v = vlist[row_idx] if row_idx < len(vlist) else None
                    if v is not None:
                        data.append({"FID series": name, "FID value": v, "alpha": a})
            x_col, y_col, color_col = "FID series", "FID value", "alpha"

        if data and pd is not None:
            df = pd.DataFrame(data)
            fig = px.line(df, x=x_col, y=y_col, color=color_col, markers=True, template="plotly_white")
            # Put legend outside on the right and leave space
            fig.update_layout(
                legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, font=dict(size=10)),
                margin=dict(r=160)
            )
            ensure_dir(out_html)
            fig.write_html(out_html, include_plotlyjs="cdn")
            print(f"Saved interactive plot to {out_html}")


if __name__ == "__main__":
    main()
