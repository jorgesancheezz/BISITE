import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.stats import gaussian_kde
from scipy.ndimage import uniform_filter1d
from scipy.stats import zscore


def load_signals(path, normalize=True):
    A = np.load(path)
    if A.ndim==3:
        A = A.reshape(A.shape[0], -1)
    elif A.ndim==1:
        A = A.reshape(1, -1)
    elif A.ndim==2 and A.shape[1]==1:
        A = A.reshape(A.shape[0], -1)
    if normalize:
        try:
            A = zscore(A, axis=1, ddof=0)
            A = np.nan_to_num(A)
        except Exception:
            pass
    return A


def mean_psd(A, fs=300.0, nperseg=512, max_signals=500):
    n = min(A.shape[0], max_signals)
    sel = A[:n]
    psds = []
    for s in sel:
        f,p = welch(s, fs=fs, nperseg=nperseg)
        psds.append(p)
    psds = np.array(psds)
    return f, psds.mean(axis=0), np.percentile(psds, 25, axis=0), np.percentile(psds,75,axis=0)


def detect_rr(sig, fs=300.0):
    y = uniform_filter1d(sig, size=5)
    peaks,props = find_peaks(y, distance=int(0.25*fs), prominence=(np.std(y)*0.4))
    if len(peaks) < 2:
        return np.array([])
    rr = np.diff(peaks)/fs
    return rr


def plot_psd_overlay(real, synth, out, fs=300.0, nperseg=512, max_signals=500):
    f_r, p_r, p25_r, p75_r = mean_psd(real, fs=fs, nperseg=nperseg, max_signals=max_signals)
    f_s, p_s, p25_s, p75_s = mean_psd(synth, fs=fs, nperseg=nperseg, max_signals=max_signals)
    plt.figure(figsize=(8,5))
    plt.semilogy(f_r, p_r, label='Real', color='C0')
    plt.fill_between(f_r, p25_r, p75_r, color='C0', alpha=0.2)
    plt.semilogy(f_s, p_s, label='Synth', color='C1')
    plt.fill_between(f_s, p25_s, p75_s, color='C1', alpha=0.2)
    max_freq = min(200.0, fs/2.0)
    plt.xlim(0, max_freq)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (log scale)')
    plt.title(f'PSD overlay (mean ± IQR) — showing 0-{int(max_freq)} Hz')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out, 'psd_overlay.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_projection_kde(real, synth, out):
    # project to first PC by simple SVD on concatenated data
    X = np.vstack([real, synth])
    Xc = X - X.mean(axis=1, keepdims=True)
    try:
        u,s,vt = np.linalg.svd(Xc, full_matrices=False)
        pc = vt[0]
    except Exception:
        pc = np.ones(X.shape[1])
    rproj = (real - real.mean(axis=1, keepdims=True)).dot(pc)
    sproj = (synth - synth.mean(axis=1, keepdims=True)).dot(pc)
    # KDE
    kr = gaussian_kde(rproj)
    ks = gaussian_kde(sproj)
    mn = min(rproj.min(), sproj.min())
    mx = max(rproj.max(), sproj.max())
    xs = np.linspace(mn, mx, 400)
    plt.figure(figsize=(7,4))
    plt.plot(xs, kr(xs), label='Real', color='C0')
    plt.plot(xs, ks(xs), label='Synth', color='C1')
    plt.fill_between(xs, kr(xs), color='C0', alpha=0.15)
    plt.fill_between(xs, ks(xs), color='C1', alpha=0.15)
    plt.title('1D projection KDE (first PC)')
    plt.xlabel('Projection')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out, 'projection_kde.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_rr_hist(real, synth, out, fs=300.0):
    # detect RR for up to 300 signals
    rrs_r = []
    rrs_s = []
    for s in real[:300]:
        rr = detect_rr(s, fs=fs)
        if rr.size>0:
            rrs_r.extend(list(rr))
    for s in synth[:300]:
        rr = detect_rr(s, fs=fs)
        if rr.size>0:
            rrs_s.extend(list(rr))
    rrs_r = np.array(rrs_r)
    rrs_s = np.array(rrs_s)
    plt.figure(figsize=(8,4))
    bins = np.linspace(0.2,1.5,60)
    plt.hist(rrs_r, bins=bins, alpha=0.6, density=True, label='Real', color='C0')
    plt.hist(rrs_s, bins=bins, alpha=0.6, density=True, label='Synth', color='C1')
    plt.xlabel('RR interval (s)')
    plt.ylabel('Density')
    plt.title('RR interval distribution (detected peaks)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out, 'rr_hist.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_mean_overlay(real, synth, out, n_examples=6):
    # plot a few example traces side by side, plus mean overlay
    n = min(real.shape[0], synth.shape[0], n_examples)
    # choose first n
    fig, axs = plt.subplots(2, n, figsize=(3*n,6), sharey=False)
    for i in range(n):
        axs[0,i].plot(real[i], color='C0')
        axs[0,i].set_title(f'Real {i}')
        axs[1,i].plot(synth[i], color='C1')
        axs[1,i].set_title(f'Synth {i}')
    plt.tight_layout()
    traces_path = os.path.join(out, 'example_traces_grid.png')
    plt.savefig(traces_path, dpi=150)
    plt.close()
    # mean overlay
    mreal = real.mean(axis=0)
    msynth = synth.mean(axis=0)
    t = np.arange(len(mreal))
    plt.figure(figsize=(8,4))
    plt.plot(t, mreal, label='Real mean', color='C0', alpha=0.9)
    plt.plot(t, msynth, label='Synth mean', color='C1', alpha=0.9)
    plt.fill_between(t, mreal, msynth, color='gray', alpha=0.15)
    plt.title('Mean signal overlay')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude (z-scored)')
    plt.legend()
    plt.tight_layout()
    mean_path = os.path.join(out, 'mean_overlay.png')
    plt.savefig(mean_path, dpi=150)
    plt.close()
    return traces_path, mean_path


def plot_hrv_violin(real, synth, out, fs=300.0):
    # compute per-signal RMSSD and SDNN for up to 200 signals
    def per_signal_hrv(A):
        vals_sdnn = []
        vals_rmssd = []
        for s in A[:200]:
            rr = detect_rr(s, fs=fs)
            if rr.size>1:
                vals_sdnn.append(np.std(rr, ddof=1))
                vals_rmssd.append(np.sqrt(np.mean(np.diff(rr)**2)))
        return np.array(vals_sdnn), np.array(vals_rmssd)
    sdnn_r, rmssd_r = per_signal_hrv(real)
    sdnn_s, rmssd_s = per_signal_hrv(synth)
    plt.figure(figsize=(8,4))
    parts = [sdnn_r, sdnn_s]
    labels = ['Real','Synth']
    plt.violinplot(parts, showmeans=True)
    plt.xticks([1,2], labels)
    plt.title('SDNN distribution (violin)')
    plt.ylabel('SDNN (s)')
    plt.tight_layout()
    v1 = os.path.join(out, 'sdnn_violin.png')
    plt.savefig(v1, dpi=150)
    plt.close()

    plt.figure(figsize=(8,4))
    parts = [rmssd_r, rmssd_s]
    plt.violinplot(parts, showmeans=True)
    plt.xticks([1,2], labels)
    plt.title('RMSSD distribution (violin)')
    plt.ylabel('RMSSD (s)')
    plt.tight_layout()
    v2 = os.path.join(out, 'rmssd_violin.png')
    plt.savefig(v2, dpi=150)
    plt.close()
    return v1, v2


def make_pretty_plots(real_path, synth_path, outdir, fs=300.0):
    os.makedirs(outdir, exist_ok=True)
    R = load_signals(real_path, normalize=True)
    S = load_signals(synth_path, normalize=True)
    print('Signals loaded:', R.shape, S.shape)
    paths = {}
    paths['psd'] = plot_psd_overlay(R, S, outdir, fs=fs)
    paths['proj'] = plot_projection_kde(R, S, outdir)
    paths['rr'] = plot_rr_hist(R, S, outdir, fs=fs)
    tgrid, meanp = plot_mean_overlay(R, S, outdir)
    paths['traces_grid'] = tgrid
    paths['mean_overlay'] = meanp
    v1,v2 = plot_hrv_violin(R, S, outdir, fs=fs)
    paths['sdnn_violin'] = v1
    paths['rmssd_violin'] = v2
    return paths

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', required=True)
    parser.add_argument('--synth', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fs', type=float, default=300.0)
    args = parser.parse_args()
    p = make_pretty_plots(args.real, args.synth, args.out, fs=args.fs)
    for k,v in p.items():
        print('WROTE', k, v)
