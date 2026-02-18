import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
import torch

try:
    # Prefer core implementation
    from PULSOVITAL.core.fid_all_in_one import (
        extract_batch_features,
        compute_stats,
        frechet_distance,
    )
except Exception:
    # Fallback: try package root shim if present
    from PULSOVITAL.fid_all_in_one import (
        extract_batch_features,
        compute_stats,
        frechet_distance,
    )


def _glob_many(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        out.extend(sorted(glob.glob(p)))
    return out


def _load_array(path: str) -> np.ndarray:
    p = path.lower()
    if p.endswith(".npy"):
        arr = np.load(path)
    elif p.endswith(".npz"):
        arr = np.load(path)["arr_0"]
    elif p.endswith(".csv"):
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = arr[..., None]  # add channel dim
    else:
        raise ValueError(f"Unsupported file type: {path}")
    # Normalize shapes to (N, T, 1)
    if arr.ndim == 1:
        arr = arr[None, :, None]
    elif arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3 and arr.shape[-1] != 1:
        # If has channels > 1, keep first channel for comparable metrics
        arr = arr[..., :1]
    return arr.astype(np.float32)


def load_dataset(paths: List[str], max_samples: int = 0) -> np.ndarray:
    files = _glob_many(paths)
    if not files:
        raise FileNotFoundError(f"No files matched: {paths}")
    arrays = []
    total = 0
    for f in files:
        a = _load_array(f)
        arrays.append(a)
        total += a.shape[0]
        if max_samples and total >= max_samples:
            break
    if not arrays:
        raise RuntimeError("No arrays loaded")
    data = np.concatenate(arrays, axis=0)
    if max_samples:
        data = data[:max_samples]
    return data  # (N, T, 1)


def time_domain_stats(x: np.ndarray) -> Dict[str, float]:
    # x: (N, T, 1)
    x2 = x[..., 0]
    per_mean = x2.mean(axis=1)
    per_std = x2.std(axis=1)
    per_min = x2.min(axis=1)
    per_max = x2.max(axis=1)
    per_rms = np.sqrt((x2 ** 2).mean(axis=1))
    # zero-crossing rate per sample
    sign = np.sign(x2)
    zcr = (sign[:, 1:] * sign[:, :-1] < 0).mean(axis=1)
    # distributional stats across all points
    flat = x2.reshape(-1)
    skew = float(stats.skew(flat, bias=False)) if flat.size > 2 else float("nan")
    kurt = float(stats.kurtosis(flat, bias=False)) if flat.size > 3 else float("nan")
    p25, p50, p75 = np.percentile(flat, [25, 50, 75])
    return {
        "samples": float(x2.shape[0]),
        "length_mean": float(x2.shape[1]),
        "mean_mean": float(per_mean.mean()),
        "mean_std": float(per_mean.std(ddof=1)) if per_mean.size > 1 else 0.0,
        "std_mean": float(per_std.mean()),
        "std_std": float(per_std.std(ddof=1)) if per_std.size > 1 else 0.0,
        "min_mean": float(per_min.mean()),
        "max_mean": float(per_max.mean()),
        "rms_mean": float(per_rms.mean()),
        "zcr_mean": float(zcr.mean()),
        "skew": float(skew),
        "kurtosis": float(kurt),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
    }


def spectral_stats(x: np.ndarray, sr: float = 1.0) -> Dict[str, float]:
    # Simple FFT-based spectral features per sample, aggregated
    x2 = x[..., 0]
    N, T = x2.shape
    # rfft magnitude
    mags = np.abs(np.fft.rfft(x2, axis=1)) + 1e-12
    freqs = np.fft.rfftfreq(T, d=1.0 / sr)
    # Centroid and bandwidth per sample
    centroid = (mags * freqs[None, :]).sum(axis=1) / mags.sum(axis=1)
    bandwidth = np.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * mags).sum(axis=1) / mags.sum(axis=1))
    # Flatness (geo/arith mean)
    geo = np.exp(np.log(mags).mean(axis=1))
    arith = mags.mean(axis=1)
    flatness = geo / arith
    return {
        "spec_centroid_mean": float(centroid.mean()),
        "spec_centroid_std": float(centroid.std(ddof=1)) if N > 1 else 0.0,
        "spec_bandwidth_mean": float(bandwidth.mean()),
        "spec_bandwidth_std": float(bandwidth.std(ddof=1)) if N > 1 else 0.0,
        "spec_flatness_mean": float(flatness.mean()),
        "spec_flatness_std": float(flatness.std(ddof=1)) if N > 1 else 0.0,
    }


def compute_fid(ref: np.ndarray, cmp: np.ndarray) -> float:
    # Use the same feature extractor used elsewhere: spectrogram -> mean over time
    ref_t = torch.from_numpy(ref)
    cmp_t = torch.from_numpy(cmp)
    ref_feats = extract_batch_features(ref_t)
    cmp_feats = extract_batch_features(cmp_t)
    mu1, sigma1 = compute_stats(ref_feats)
    mu2, sigma2 = compute_stats(cmp_feats)
    return float(frechet_distance(mu1, sigma1, mu2, sigma2))


def compare(ref_paths: List[str], cmp_paths: List[str], max_samples: int = 0) -> Dict[str, Dict[str, float]]:
    ref = load_dataset(ref_paths, max_samples=max_samples)
    cmp = load_dataset(cmp_paths, max_samples=max_samples)

    ref_time = time_domain_stats(ref)
    cmp_time = time_domain_stats(cmp)

    ref_spec = spectral_stats(ref)
    cmp_spec = spectral_stats(cmp)

    fid = compute_fid(ref, cmp)

    return {
        "ref_time": ref_time,
        "cmp_time": cmp_time,
        "ref_spec": ref_spec,
        "cmp_spec": cmp_spec,
        "fid": {"spectrogram_fid": fid},
    }


def main():
    ap = argparse.ArgumentParser(description="Compare important statistics between two synthetic datasets")
    ap.add_argument("--ref", nargs="+", required=True, help="Reference file(s) or glob(s)")
    ap.add_argument("--cmp", nargs="+", required=True, help="Comparison file(s) or glob(s)")
    ap.add_argument("--max-samples", type=int, default=0, help="Limit number of samples to load (0 = all)")
    ap.add_argument("--out-json", type=str, default="", help="Optional JSON output path")
    args = ap.parse_args()

    report = compare(args.ref, args.cmp, max_samples=args.max_samples)

    # Pretty print
    print("=== TIME DOMAIN (REF) ===")
    for k, v in report["ref_time"].items():
        print(f"{k:20s}: {v}")
    print("\n=== TIME DOMAIN (CMP) ===")
    for k, v in report["cmp_time"].items():
        print(f"{k:20s}: {v}")
    print("\n=== SPECTRAL (REF) ===")
    for k, v in report["ref_spec"].items():
        print(f"{k:20s}: {v}")
    print("\n=== SPECTRAL (CMP) ===")
    for k, v in report["cmp_spec"].items():
        print(f"{k:20s}: {v}")
    print("\n=== DISTANCE ===")
    print(f"spectrogram_FID    : {report['fid']['spectrogram_fid']}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved JSON report to {args.out_json}")


if __name__ == "__main__":
    main()
