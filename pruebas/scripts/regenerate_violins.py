import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Config
FS = 300.0
MAX_SAMPLE = 1024
SEED = 42

paths = {
  'AF_real': 'PULSOVITAL/npy_output/AF_processed_1024x3000x1.npy',
  'AF_synth': 'PULSOVITAL/Metricas/1024seq_AF.npy',
  'NSR_real': 'PULSOVITAL/npy_output/NSR_processed_1024x3000x1.npy',
  'NSR_synth': 'PULSOVITAL/Metricas/1024seq_NSR.npy'
}

def prepare(A):
    A = np.asarray(A)
    if A.ndim==3:
        A = A.reshape(A.shape[0], -1)
    elif A.ndim==1:
        A = A.reshape(1, -1)
    elif A.ndim==2 and A.shape[1]==1:
        A = A.reshape(A.shape[0], -1)
    A = (A - A.mean(axis=1, keepdims=True)) / (A.std(axis=1, keepdims=True) + 1e-8)
    return A


def compute_rmssd_sdnn_per_sample(A, max_samples=MAX_SAMPLE):
    r = []
    s = []
    for sig in A[:max_samples]:
        sig = np.asarray(sig).ravel()
        if len(sig) < 2:
            r.append(np.nan); s.append(np.nan); continue
        diffs = np.diff(sig)
        rmssd = np.sqrt(np.mean(diffs**2))
        sdnn = float(np.std(sig))
        r.append(rmssd); s.append(sdnn)
    return np.array(r), np.array(s)


def plot_violins(rr_R, rr_S, sd_R, sd_S, outdir):
    os.makedirs(outdir, exist_ok=True)
    try:
        import seaborn as sns
        import pandas as pd
        df_rm = pd.DataFrame({'value': np.concatenate([rr_R, rr_S]), 'class': ['Real']*len(rr_R) + ['Synth']*len(rr_S)})
        df_sd = pd.DataFrame({'value': np.concatenate([sd_R, sd_S]), 'class': ['Real']*len(sd_R) + ['Synth']*len(sd_S)})
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        sns.violinplot(x='class', y='value', data=df_rm, palette=['C0','C1'])
        plt.title('RMSSD por clase')
        plt.subplot(1,2,2)
        sns.violinplot(x='class', y='value', data=df_sd, palette=['C0','C1'])
        plt.title('SDNN por clase')
        plt.tight_layout()
        outp = os.path.join(outdir, 'rmssd_sdnn_violin.png')
        plt.savefig(outp, dpi=150); plt.close()
        # save separate files
        plt.figure(figsize=(5,4))
        sns.violinplot(x='class', y='value', data=df_rm, palette=['C0','C1'])
        plt.title('RMSSD por clase')
        plt.tight_layout(); plt.savefig(os.path.join(outdir,'rmssd_violin.png'), dpi=150); plt.close()
        plt.figure(figsize=(5,4))
        sns.violinplot(x='class', y='value', data=df_sd, palette=['C0','C1'])
        plt.title('SDNN por clase')
        plt.tight_layout(); plt.savefig(os.path.join(outdir,'sdnn_violin.png'), dpi=150); plt.close()
        return True
    except Exception:
        # fallback to boxplots
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.boxplot([rr_R[~np.isnan(rr_R)], rr_S[~np.isnan(rr_S)]], labels=['Real','Synth'])
        plt.title('RMSSD por clase')
        plt.subplot(1,2,2)
        plt.boxplot([sd_R[~np.isnan(sd_R)], sd_S[~np.isnan(sd_S)]], labels=['Real','Synth'])
        plt.title('SDNN por clase')
        plt.tight_layout()
        outp = os.path.join(outdir, 'rmssd_sdnn_box.png')
        plt.savefig(outp, dpi=150); plt.close()
        return False


def load_or_fail(p):
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p)


def main():
    try:
        AFR = load_or_fail(paths['AF_real'])
        AFS = load_or_fail(paths['AF_synth'])
        NSRR = load_or_fail(paths['NSR_real'])
        NSRS = load_or_fail(paths['NSR_synth'])
    except Exception as e:
        print('Error loading npy files:', e)
        return
    AFR_p = prepare(AFR)
    AFS_p = prepare(AFS)
    NSRR_p = prepare(NSRR)
    NSRS_p = prepare(NSRS)

    # AF
    rr_R, sd_R = compute_rmssd_sdnn_per_sample(AFR_p)
    rr_S, sd_S = compute_rmssd_sdnn_per_sample(AFS_p)
    ok_af = plot_violins(rr_R, rr_S, sd_R, sd_S, 'compare_out_pretty_AF')
    print('AF violins generated:', ok_af)

    # NSR
    rr_R2, sd_R2 = compute_rmssd_sdnn_per_sample(NSRR_p)
    rr_S2, sd_S2 = compute_rmssd_sdnn_per_sample(NSRS_p)
    ok_nsr = plot_violins(rr_R2, rr_S2, sd_R2, sd_S2, 'compare_out_pretty_NSR')
    print('NSR violins generated:', ok_nsr)

if __name__ == '__main__':
    main()
