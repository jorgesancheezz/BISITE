"""Generate exhaustive comparison plots for AF and NSR.

This script mirrors the plotting utilities from the notebook and produces a
wide set of PNGs into `compare_out_pretty_AF/` and `compare_out_pretty_NSR/`.
"""
import os
import numpy as np
from scipy.signal import welch, find_peaks
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Parameters (match notebook)
FS = 300.0
MAX_SAMPLE = 500
PSD_MAX_FREQ = min(200.0, FS/2.0)
SEED = 42

paths = {
    'AF_real': 'PULSOVITAL/npy_output/AF_processed_1024x3000x1.npy',
    'AF_synth': 'PULSOVITAL/Metricas/1024seq_AF.npy',
    'NSR_real': 'PULSOVITAL/npy_output/NSR_processed_1024x3000x1.npy',
    'NSR_synth': 'PULSOVITAL/Metricas/1024seq_NSR.npy',
}

def prepare(A):
    A = np.asarray(A)
    if A.ndim == 3:
        A = A.reshape(A.shape[0], -1)
    elif A.ndim == 1:
        A = A.reshape(1, -1)
    elif A.ndim == 2 and A.shape[1] == 1:
        A = A.reshape(A.shape[0], -1)
    # z-score per signal
    A = (A - A.mean(axis=1, keepdims=True)) / (A.std(axis=1, keepdims=True) + 1e-8)
    return A

def ensure_outdir(d):
    os.makedirs(d, exist_ok=True)

def mean_psd(A, nperseg=1024, nmax=MAX_SAMPLE):
    ps = []
    for s in A[:min(len(A), nmax)]:
        f,p = welch(s, fs=FS, nperseg=nperseg)
        ps.append(p)
    ps = np.array(ps)
    return f, np.mean(ps, axis=0), np.percentile(ps,25,axis=0), np.percentile(ps,75,axis=0), ps

def save_psd_overlay(R, S, outdir, title):
    ensure_outdir(outdir)
    f_r, p_r, p25r, p75r, _ = mean_psd(R)
    f_s, p_s, p25s, p75s, _ = mean_psd(S)
    idx = f_r <= PSD_MAX_FREQ
    plt.figure(figsize=(8,4))
    plt.semilogy(f_r[idx], p_r[idx], label='Real', color='C0')
    plt.semilogy(f_s[idx], p_s[idx], label='Synth', color='C1')
    plt.fill_between(f_r[idx], p25r[idx], p75r[idx], color='C0', alpha=0.15)
    plt.fill_between(f_s[idx], p25s[idx], p75s[idx], color='C1', alpha=0.15)
    plt.xlim(0, PSD_MAX_FREQ)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (log)')
    plt.title(title + ' — PSD overlay')
    plt.legend()
    plt.tight_layout()
    pth = os.path.join(outdir,'psd_overlay.png')
    plt.savefig(pth, dpi=150)
    plt.close()

def save_per_sample_psd_grid(A, outdir, prefix, n_samples=36):
    ensure_outdir(outdir)
    n = min(n_samples, len(A))
    cols = 6
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=(cols*2, rows*1.5))
    for i in range(n):
        f,p = welch(A[i], fs=FS, nperseg=1024)
        idx = f <= PSD_MAX_FREQ
        plt.subplot(rows, cols, i+1)
        plt.semilogy(f[idx], p[idx], color='C0')
        plt.xticks([]); plt.yticks([])
    plt.suptitle(prefix + ' — per-sample PSD')
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(os.path.join(outdir, f'{prefix}_per_sample_psd.png'), dpi=150)
    plt.close()

def save_psd_heatmap(A, outdir, prefix, nmax=200):
    ensure_outdir(outdir)
    psds = []
    for s in A[:min(nmax,len(A))]:
        f,p = welch(s, fs=FS, nperseg=1024)
        psds.append(p)
    psds = np.array(psds)
    plt.figure(figsize=(8,4))
    f_right = min(f[-1], PSD_MAX_FREQ)
    plt.imshow(10*np.log10(psds+1e-12), aspect='auto', origin='lower', extent=[f[0], f_right, 0, psds.shape[0]])
    plt.colorbar(label='PSD (dB)'); plt.xlabel('Frequency (Hz)'); plt.title(prefix + ' — PSD heatmap'); plt.xlim(0, PSD_MAX_FREQ); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{prefix}_psd_heatmap.png'), dpi=150)
    plt.close()

def save_mean_waveform(R, S, outdir, title, n_vis=100):
    ensure_outdir(outdir)
    n_vis = min(n_vis, len(R), len(S))
    mean_R = np.mean(R[:n_vis], axis=0)
    std_R = np.std(R[:n_vis], axis=0)
    mean_S = np.mean(S[:n_vis], axis=0)
    std_S = np.std(S[:n_vis], axis=0)
    plt.figure(figsize=(10,4))
    t = np.arange(mean_R.size)/FS
    plt.plot(t, mean_R, label='Real', color='C0')
    plt.fill_between(t, mean_R-std_R, mean_R+std_R, color='C0', alpha=0.2)
    plt.plot(t, mean_S, label='Synth', color='C1')
    plt.fill_between(t, mean_S-std_S, mean_S+std_S, color='C1', alpha=0.2)
    plt.xlabel('Time (s)'); plt.title(title + ' — Mean ± STD'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,'mean_waveform.png'), dpi=150); plt.close()

def autocorr(x, max_lag=None):
    x = x - np.mean(x)
    N = len(x)
    corr = np.correlate(x, x, mode='full')
    corr = corr[N-1:]
    denom = (np.var(x) * np.arange(N, 0, -1))
    denom[denom==0]=1.0
    corr = corr / denom
    if max_lag is None:
        return corr
    return corr[:max_lag]

def save_mean_autocorr(R, S, outdir, title, max_lag=None):
    ensure_outdir(outdir)
    if max_lag is None:
        max_lag = min(500, R.shape[1]//2)
    acs_r = [autocorr(x, max_lag=max_lag) for x in R[:min(len(R),100)]]
    acs_s = [autocorr(x, max_lag=max_lag) for x in S[:min(len(S),100)]]
    ac_r = np.mean(acs_r, axis=0)
    ac_s = np.mean(acs_s, axis=0)
    lags = np.arange(len(ac_r))/FS
    plt.figure(figsize=(8,4))
    plt.plot(lags, ac_r, label='Real', color='C0')
    plt.plot(lags, ac_s, label='Synth', color='C1')
    plt.xlabel('Lag (s)'); plt.title(title + ' — Mean Autocorr'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,'mean_autocorr.png'), dpi=150); plt.close()

def save_pca_scatter(R, S, outdir, title):
    ensure_outdir(outdir)
    try:
        X = np.vstack([R[:MAX_SAMPLE], S[:MAX_SAMPLE]])
        y = np.hstack([np.zeros(len(R[:MAX_SAMPLE])), np.ones(len(S[:MAX_SAMPLE]))])
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(X)
        plt.figure(figsize=(6,5))
        plt.scatter(Xp[y==0,0], Xp[y==0,1], s=8, alpha=0.6, label='Real', color='C0')
        plt.scatter(Xp[y==1,0], Xp[y==1,1], s=8, alpha=0.6, label='Synth', color='C1')
        plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title(title + ' — PCA scatter'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir,'pca_scatter.png'), dpi=150); plt.close()
    except Exception:
        pass

def save_discriminator_roc(R, S, outdir, title):
    ensure_outdir(outdir)
    try:
        X = np.vstack([R[:MAX_SAMPLE], S[:MAX_SAMPLE]])
        y = np.hstack([np.zeros(len(R[:MAX_SAMPLE])), np.ones(len(S[:MAX_SAMPLE]))])
        pca = PCA(n_components=min(50, X.shape[1]))
        Xf = pca.fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.3, random_state=SEED, stratify=y)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, ytr)
        probs = clf.predict_proba(Xte)[:,1]
        fpr, tpr, _ = roc_curve(yte, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label='AUC={:.3f}'.format(roc_auc))
        plt.plot([0,1],[0,1],'k--',alpha=0.3)
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(title + ' — Discriminator ROC'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir,'discriminator_roc.png'), dpi=150); plt.close()
    except Exception:
        pass

def save_sample_traces_grid(R, S, outdir, title, n_grid=6):
    ensure_outdir(outdir)
    n_vis = min(50, len(R), len(S))
    n_grid = min(n_grid, n_vis)
    plt.figure(figsize=(12,6))
    for i in range(n_grid):
        plt.subplot(2, n_grid, i+1)
        plt.plot(np.arange(R[i].size)/FS, R[i], color='C0')
        plt.title('Real')
        plt.xticks([]); plt.yticks([])
        plt.subplot(2, n_grid, n_grid + i+1)
        plt.plot(np.arange(S[i].size)/FS, S[i], color='C1')
        plt.title('Synth')
        plt.xticks([]); plt.yticks([])
    plt.suptitle(title + ' — Sample traces')
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(os.path.join(outdir,'sample_traces_grid.png'), dpi=150); plt.close()

def save_skew_kurt_boxplots(R, S, outdir, title):
    ensure_outdir(outdir)
    sk_r = skew(R, axis=1); sk_s = skew(S, axis=1)
    ku_r = kurtosis(R, axis=1); ku_s = kurtosis(S, axis=1)
    plt.figure(figsize=(6,4))
    plt.boxplot([sk_r, sk_s], tick_labels=['Real','Synth'])
    plt.title(title + ' — Skewness boxplot'); plt.tight_layout()
    plt.savefig(os.path.join(outdir,'skewness_boxplot.png'), dpi=150); plt.close()
    plt.figure(figsize=(6,4))
    plt.boxplot([ku_r, ku_s], tick_labels=['Real','Synth'])
    plt.title(title + ' — Kurtosis boxplot'); plt.tight_layout()
    plt.savefig(os.path.join(outdir,'kurtosis_boxplot.png'), dpi=150); plt.close()

def save_rr_hist(R, S, outdir, title):
    ensure_outdir(outdir)
    rrs_r = []
    rrs_s = []
    for s in R[:min(len(R),300)]:
        peaks,_ = find_peaks(s, distance=int(0.25*FS), prominence=np.std(s)*0.4)
        if len(peaks)>1:
            rrs_r.extend(np.diff(peaks)/FS)
    for s in S[:min(len(S),300)]:
        peaks,_ = find_peaks(s, distance=int(0.25*FS), prominence=np.std(s)*0.4)
        if len(peaks)>1:
            rrs_s.extend(np.diff(peaks)/FS)
    plt.figure(figsize=(8,4))
    bins=np.linspace(0.2,1.5,60)
    plt.hist(rrs_r, bins=bins, density=True, alpha=0.6, color='C0', label='Real')
    plt.hist(rrs_s, bins=bins, density=True, alpha=0.6, color='C1', label='Synth')
    plt.title(title + ' — RR histogram')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,'rr_hist.png'), dpi=150); plt.close()

def generate_all():
    # load
    AFR = np.load(paths['AF_real'])
    AFS = np.load(paths['AF_synth'])
    NSRR = np.load(paths['NSR_real'])
    NSRS = np.load(paths['NSR_synth'])
    AFR_p = prepare(AFR)
    AFS_p = prepare(AFS)
    NSRR_p = prepare(NSRR)
    NSRS_p = prepare(NSRS)

    # AF
    out_af = 'compare_out_pretty_AF'
    save_psd_overlay(AFR_p, AFS_p, out_af, 'AF processed vs synth')
    save_per_sample_psd_grid(AFR_p, out_af, 'AF_real')
    save_per_sample_psd_grid(AFS_p, out_af, 'AF_synth')
    save_psd_heatmap(AFR_p, out_af, 'AF_real')
    save_psd_heatmap(AFS_p, out_af, 'AF_synth')
    save_mean_waveform(AFR_p, AFS_p, out_af, 'AF processed vs synth')
    save_mean_autocorr(AFR_p, AFS_p, out_af, 'AF processed vs synth')
    save_pca_scatter(AFR_p, AFS_p, out_af, 'AF processed vs synth')
    save_discriminator_roc(AFR_p, AFS_p, out_af, 'AF processed vs synth')
    save_sample_traces_grid(AFR_p, AFS_p, out_af, 'AF processed vs synth')
    save_skew_kurt_boxplots(AFR_p, AFS_p, out_af, 'AF processed vs synth')
    save_rr_hist(AFR_p, AFS_p, out_af, 'AF processed vs synth')

    # NSR
    out_nsr = 'compare_out_pretty_NSR'
    save_psd_overlay(NSRR_p, NSRS_p, out_nsr, 'NSR processed vs synth')
    save_per_sample_psd_grid(NSRR_p, out_nsr, 'NSR_real')
    save_per_sample_psd_grid(NSRS_p, out_nsr, 'NSR_synth')
    save_psd_heatmap(NSRR_p, out_nsr, 'NSR_real')
    save_psd_heatmap(NSRS_p, out_nsr, 'NSR_synth')
    save_mean_waveform(NSRR_p, NSRS_p, out_nsr, 'NSR processed vs synth')
    save_mean_autocorr(NSRR_p, NSRS_p, out_nsr, 'NSR processed vs synth')
    save_pca_scatter(NSRR_p, NSRS_p, out_nsr, 'NSR processed vs synth')
    save_discriminator_roc(NSRR_p, NSRS_p, out_nsr, 'NSR processed vs synth')
    save_sample_traces_grid(NSRR_p, NSRS_p, out_nsr, 'NSR processed vs synth')
    save_skew_kurt_boxplots(NSRR_p, NSRS_p, out_nsr, 'NSR processed vs synth')
    save_rr_hist(NSRR_p, NSRS_p, out_nsr, 'NSR processed vs synth')

    print('All plots generated.')

if __name__ == '__main__':
    generate_all()
