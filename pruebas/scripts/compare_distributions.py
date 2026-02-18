#!/usr/bin/env python3
"""Compare PSD and RR-derived metrics between real and synthetic p09 datasets.

Outputs plots and a short markdown report in `compare_out_test/`.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.stats import ks_2samp, wasserstein_distance
import json


def load_signals(path):
    a = np.load(path)
    if a.ndim == 3:
        a = a.reshape(a.shape[0], a.shape[1])
    return a


def compute_mean_psd(X, fs=250.0, nperseg=512):
    psds = []
    freqs = None
    for x in X:
        f, Pxx = welch(x, fs=fs, nperseg=nperseg)
        psds.append(Pxx)
        freqs = f
    psds = np.array(psds)
    return freqs, psds, psds.mean(axis=0)


def detect_rpeaks(sig, fs=250.0):
    # naive R-peak detection on a single-lead signal: find peaks on absolute derivative
    s = sig - np.mean(sig)
    d = np.abs(np.diff(s))
    # find peaks in derivative
    peaks, _ = find_peaks(d, distance=fs*0.2, height=np.std(d)*0.5)
    # convert index in diff to index in original
    peaks = np.clip(peaks + 1, 0, len(sig)-1)
    return peaks


def rr_metrics_from_signal(sig, fs=250.0):
    peaks = detect_rpeaks(sig, fs=fs)
    if len(peaks) < 4:
        return np.nan, np.nan
    rr = np.diff(peaks) / float(fs)
    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2)))
    mean_rr = float(np.mean(rr)) if np.mean(rr) > 0 else np.nan
    cv = float(np.std(rr) / mean_rr) if mean_rr > 0 else np.nan
    return rmssd, cv


def compute_rr_stats(X, fs=250.0):
    rms = []
    cvs = []
    for x in X:
        r,c = rr_metrics_from_signal(x, fs=fs)
        rms.append(r)
        cvs.append(c)
    return np.array(rms), np.array(cvs)


def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def main():
    outd = Path('compare_out_test')
    safe_mkdir(outd)

    real_af_p = Path('PULSOVITAL/npy_output_p09_consolidated/1024seq_AF.npy')
    real_nsr_p = Path('PULSOVITAL/npy_output_p09_consolidated/1024seq_NSR.npy')
    synth_af_p = Path('1024seq_AF.npy')
    synth_nsr_p = Path('1024seq_NSR.npy')

    for p in (real_af_p, real_nsr_p, synth_af_p, synth_nsr_p):
        if not p.exists():
            raise FileNotFoundError(p)

    X_real_af = load_signals(real_af_p)
    X_real_nsr = load_signals(real_nsr_p)
    X_synth_af = load_signals(synth_af_p)
    X_synth_nsr = load_signals(synth_nsr_p)

    # PSD
    fs = 250.0
    f_ra, psd_ra, mean_ra = compute_mean_psd(X_real_af, fs=fs)
    f_rn, psd_rn, mean_rn = compute_mean_psd(X_real_nsr, fs=fs)
    f_sa, psd_sa, mean_sa = compute_mean_psd(X_synth_af, fs=fs)
    f_sn, psd_sn, mean_sn = compute_mean_psd(X_synth_nsr, fs=fs)

    # plot mean PSDs
    plt.figure(figsize=(8,5))
    plt.semilogy(f_ra, mean_ra, label='real AF')
    plt.semilogy(f_rn, mean_rn, label='real NSR')
    plt.semilogy(f_sa, mean_sa, label='synth AF')
    plt.semilogy(f_sn, mean_sn, label='synth NSR')
    plt.xlim(0,50)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(True)
    p1 = outd / 'mean_psd.png'
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()

    # RR metrics
    rms_ra, cvs_ra = compute_rr_stats(X_real_af, fs=fs)
    rms_rn, cvs_rn = compute_rr_stats(X_real_nsr, fs=fs)
    rms_sa, cvs_sa = compute_rr_stats(X_synth_af, fs=fs)
    rms_sn, cvs_sn = compute_rr_stats(X_synth_nsr, fs=fs)

    # compute KS and Wasserstein between distributions (real vs synth) per class
    stats = {}
    for cls, real_rms, synth_rms, real_cvs, synth_cvs in (
        ('AF', rms_ra, rms_sa, cvs_ra, cvs_sa),
        ('NSR', rms_rn, rms_sn, cvs_rn, cvs_sn),
    ):
        # remove NaN
        rr_real = real_rms[np.isfinite(real_rms)]
        rr_synth = synth_rms[np.isfinite(synth_rms)]
        cv_real = real_cvs[np.isfinite(real_cvs)]
        cv_synth = synth_cvs[np.isfinite(synth_cvs)]
        stats[cls] = {
            'rmssd_ks_p': float(ks_2samp(rr_real, rr_synth).pvalue) if len(rr_real)>0 and len(rr_synth)>0 else None,
            'rmssd_wass': float(wasserstein_distance(rr_real, rr_synth)) if len(rr_real)>0 and len(rr_synth)>0 else None,
            'cv_ks_p': float(ks_2samp(cv_real, cv_synth).pvalue) if len(cv_real)>0 and len(cv_synth)>0 else None,
            'cv_wass': float(wasserstein_distance(cv_real, cv_synth)) if len(cv_real)>0 and len(cv_synth)>0 else None,
        }

    # save rr hist plots
    for name, real, synth, metric in (
        ('rmssd_AF', rms_ra, rms_sa, 'RMSSD (s)'),
        ('rmssd_NSR', rms_rn, rms_sn, 'RMSSD (s)'),
        ('cv_AF', cvs_ra, cvs_sa, 'CV (unitless)'),
        ('cv_NSR', cvs_rn, cvs_sn, 'CV (unitless)'),
    ):
        plt.figure(figsize=(6,4))
        plt.hist(real[np.isfinite(real)], bins=50, alpha=0.6, density=True, label='real')
        plt.hist(synth[np.isfinite(synth)], bins=50, alpha=0.6, density=True, label='synth')
        plt.legend()
        plt.title(name)
        plt.xlabel(metric)
        plt.tight_layout()
        plt.savefig(outd / f'{name}.png')
        plt.close()

    # write JSON summary
    summary = {
        'psd_mean_files': str(p1),
        'rr_stats': stats,
        'counts': {
            'real_af': int(X_real_af.shape[0]),
            'real_nsr': int(X_real_nsr.shape[0]),
            'synth_af': int(X_synth_af.shape[0]),
            'synth_nsr': int(X_synth_nsr.shape[0]),
        }
    }
    with open(outd / 'dist_comparison_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    # short markdown report
    with open(outd / 'report.md', 'w', encoding='utf-8') as fh:
        fh.write('# Comparison report\n')
        fh.write('\n')
        fh.write('Generated plots: mean_psd.png, rmssd_AF.png, rmssd_NSR.png, cv_AF.png, cv_NSR.png\n')
        fh.write('\n')
        fh.write('## RR distribution tests (real vs synth)\n')
        for cls in stats:
            s = stats[cls]
            fh.write(f'- {cls}: RMSSD KS p={s["rmssd_ks_p"]:.4f}  RMSSD Wasserstein={s["rmssd_wass"]:.4f}\n')
            fh.write(f'         CV KS p={s["cv_ks_p"]:.4f}  CV Wasserstein={s["cv_wass"]:.4f}\n')
        fh.write('\n')
        fh.write('Note: detection of R-peaks is naive; for clinical-grade RR metrics use a robust detector.\n')

    print('Outputs written to', outd)


if __name__ == '__main__':
    main()
