import os
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.signal import welch, correlate
from scipy.stats import ks_2samp, wasserstein_distance, skew, kurtosis
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT_DIR = 'compare_out_demo'
os.makedirs(OUT_DIR, exist_ok=True)

def prepare(arr):
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a.reshape(a.shape[0], a.shape[1])
    if a.ndim == 1:
        a = a.reshape(1, -1)
    # sanitize NaN/Inf values
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a

def resample_to(a, target_len):
    # a: (N, T)
    N, T = a.shape
    if T == target_len:
        return a
    xp = np.arange(T)
    x_new = np.linspace(0, T - 1, target_len)
    out = np.zeros((N, target_len), dtype=np.float32)
    for i in range(N):
        out[i] = np.interp(x_new, xp, a[i]).astype(np.float32)
    return out

def median_rbf_sigma(X, Y):
    X = np.asarray(X).reshape(len(X), -1)
    Y = np.asarray(Y).reshape(len(Y), -1)
    Z = np.vstack([X, Y])
    m = Z.shape[0]
    inds = np.random.choice(m, size=min(m,2000), replace=False)
    sub = Z[inds]
    d = cdist(sub, sub, 'euclidean')
    med = np.median(d[np.triu_indices_from(d,1)])
    return float(med) if med>0 else 1.0

def rbf_mmd2(X, Y, sigma):
    X = np.asarray(X).reshape(len(X), -1)
    Y = np.asarray(Y).reshape(len(Y), -1)
    Kxx = np.exp(-cdist(X,X,'sqeuclidean')/(2*sigma**2))
    Kyy = np.exp(-cdist(Y,Y,'sqeuclidean')/(2*sigma**2))
    Kxy = np.exp(-cdist(X,Y,'sqeuclidean')/(2*sigma**2))
    m = X.shape[0]; n = Y.shape[0]
    sum_x = (np.sum(Kxx) - np.sum(np.diag(Kxx))) / (m*(m-1)) if m>1 else 0.0
    sum_y = (np.sum(Kyy) - np.sum(np.diag(Kyy))) / (n*(n-1)) if n>1 else 0.0
    sum_xy = np.sum(Kxy) / (m*n)
    return float(sum_x + sum_y - 2*sum_xy)

def energy_distance_flat(X, Y):
    X = np.asarray(X).ravel(); Y = np.asarray(Y).ravel()
    try:
        from scipy import stats
        return float(stats.energy_distance(X, Y))
    except Exception:
        return float('nan')

def first_pc_proj(X):
    X = np.asarray(X).reshape(len(X), -1)
    Xc = X - X.mean(axis=1, keepdims=True)
    try:
        u,s,vt = np.linalg.svd(Xc, full_matrices=False)
        pc = vt[0]
        return Xc.dot(pc)
    except Exception:
        return X.mean(axis=1)

def mean_psd(A, fs=300.0, nperseg=1024):
    ps = []
    for s in A:
        f,p = welch(s, fs=fs, nperseg=min(nperseg, len(s)))
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        ps.append(p)
    return f, np.mean(ps, axis=0)

def discriminator_auc(R, S):
    try:
        X = np.vstack([R, S])
        y = np.array([0]*len(R) + [1]*len(S))
        pca = PCA(n_components=min(50, X.shape[1]))
        Xr = pca.fit_transform(X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        clf = LogisticRegression(max_iter=2000)
        for tr,te in cv.split(Xr,y):
            if len(np.unique(y[te]))<2: continue
            clf.fit(Xr[tr], y[tr])
            probs = clf.predict_proba(Xr[te])[:,1]
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(y[te], probs))
        return float(np.mean(aucs)) if aucs else float('nan')
    except Exception:
        return float('nan')

def compute_fid_from_feats(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    # align feature dimensionality: if different, project both to common subspace via PCA
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    dim_x = X.shape[1] if X.size else 0
    dim_y = Y.shape[1] if Y.size else 0
    if dim_x == 0 or dim_y == 0:
        return float('nan')
    if dim_x != dim_y:
        try:
            from sklearn.decomposition import PCA
            k = min(dim_x, dim_y)
            Z = np.vstack([X, Y])
            pca = PCA(n_components=k)
            Zp = pca.fit_transform(Z)
            X = Zp[:len(X)]
            Y = Zp[len(X):]
        except Exception:
            # fallback: truncate/pad
            k = min(dim_x, dim_y)
            X = X[:, :k]
            Y = Y[:, :k]
    mu_x = X.mean(axis=0); mu_y = Y.mean(axis=0)
    cov_x = np.cov(X, rowvar=False); cov_y = np.cov(Y, rowvar=False)
    diff = mu_x - mu_y
    try:
        covmean = sqrtm(cov_x.dot(cov_y))
    except Exception:
        eps = 1e-6
        covmean = sqrtm((cov_x + np.eye(cov_x.shape[0])*eps).dot(cov_y + np.eye(cov_y.shape[0])*eps))
    if np.iscomplexobj(covmean): covmean = covmean.real
    fid = diff.dot(diff) + np.trace(cov_x + cov_y - 2*covmean)
    return float(np.real(fid))

def extract_psd_feats(A, fs=300.0, nperseg=1024, n_comp=64, max_samples=1024):
    X = []
    for s in A[:max_samples]:
        f,p = welch(s, fs=fs, nperseg=min(nperseg, len(s)))
        idx = f <= fs/2.0
        p = np.nan_to_num(p, nan=1e-12, posinf=1e-12, neginf=1e-12)
        X.append(np.log10(p[idx] + 1e-12))
    X = np.array(X)
    if X.shape[0] < 2:
        return np.zeros((0, min(n_comp, X.shape[1] if X.ndim>1 else 0)))
    n_comp = min(n_comp, X.shape[1], X.shape[0])
    if n_comp < X.shape[1]:
        pca = PCA(n_components=n_comp)
        X = pca.fit_transform(X)
    return X

def avg_dtw_between_sets(R, S, n_pairs=100):
    from math import ceil
    def dtw_distance(a,b,max_len=300):
        a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
        ka = max(1, ceil(len(a)/max_len)); kb = max(1, ceil(len(b)/max_len))
        a_ds = a[::ka]; b_ds = b[::kb]
        na = len(a_ds); nb = len(b_ds)
        D = np.full((na+1, nb+1), np.inf); D[0,0]=0.0
        for i in range(1,na+1):
            for j in range(1,nb+1):
                cost = abs(float(a_ds[i-1]) - float(b_ds[j-1]))
                D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        return float(D[na,nb] / max(1.0, float(na+nb)))
    n = min(len(R), len(S))
    if n==0: return float('nan')
    idxs = np.linspace(0, n-1, min(n, n_pairs)).astype(int)
    vals = []
    for i in idxs:
        try:
            vals.append(dtw_distance(R[i], S[i]))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else float('nan')

def compute_all_metrics(R, S, name='comparison'):
    # R,S are arrays (N,T)
    res = {}
    # MMD
    sigma = median_rbf_sigma(R, S)
    res['mmd2'] = rbf_mmd2(R, S, sigma)
    res['mmd_sigma'] = sigma
    # energy
    res['energy'] = energy_distance_flat(R, S)
    # first pc proj
    rproj = first_pc_proj(R)
    sproj = first_pc_proj(S)
    ks_stat, ks_p = ks_2samp(rproj, sproj)
    res['ks_stat'] = float(ks_stat); res['ks_p'] = float(ks_p)
    res['wasserstein'] = float(wasserstein_distance(rproj, sproj))
    res['mean_diff_proj'] = float(np.mean(rproj) - np.mean(sproj))
    # PSD L2
    f_r, p_r = mean_psd(R)
    f_s, p_s = mean_psd(S)
    res['psd_l2'] = float(np.linalg.norm(p_r - p_s))
    # discriminator auc
    res['discriminator_auc'] = discriminator_auc(R, S)
    # DS: discriminative score (use discriminator AUC)
    res['DS'] = res['discriminator_auc']

    # PS: predictive score — train a classifier on PSD features (fallback to PCA of signals)
    def compute_predictive_score(R, S):
        try:
            # extract PSD features
            Xr = extract_psd_feats(R, n_comp=32, max_samples=1024)
            Xs = extract_psd_feats(S, n_comp=32, max_samples=1024)
            if Xr.shape[0] >= 2 and Xs.shape[0] >= 2:
                X = np.vstack([Xr, Xs])
            else:
                # fallback: flatten signals and reduce with PCA
                XR = np.asarray(R).reshape(len(R), -1)
                XS = np.asarray(S).reshape(len(S), -1)
                # limit rows
                m = min(1024, XR.shape[0] + XS.shape[0])
                Xall = np.vstack([XR, XS])
                n_comp = min(32, Xall.shape[1])
                if n_comp <= 0 or Xall.shape[0] < 2:
                    return float('nan')
                pca = PCA(n_components=n_comp)
                X = pca.fit_transform(Xall)
            y = np.array([0]*len(Xr) + [1]*len(Xs)) if Xr.shape[0] >= 1 and Xs.shape[0] >= 1 else None
            if y is None or len(np.unique(y))<2:
                return float('nan')
            # classifier CV AUC
            cv = StratifiedKFold(n_splits=min(5, max(2, len(y)//2)), shuffle=True, random_state=42)
            aucs = []
            clf = LogisticRegression(max_iter=2000)
            for tr, te in cv.split(X, y):
                if len(np.unique(y[te]))<2: continue
                clf.fit(X[tr], y[tr])
                probs = clf.predict_proba(X[te])[:, 1]
                aucs.append(roc_auc_score(y[te], probs))
            return float(np.mean(aucs)) if aucs else float('nan')
        except Exception:
            return float('nan')

    res['PS'] = compute_predictive_score(R, S)
    # marginal distribution (Wasserstein on values)
    res['MDD'] = float(wasserstein_distance(R.ravel(), S.ravel()))
    # autocorr, skew, kurt
    def autocorr(x, max_lag=None):
        x = x - np.mean(x)
        N = len(x)
        corr = correlate(x, x, mode='full')[N-1:]
        denom = (np.var(x) * np.arange(N, 0, -1))
        denom[denom==0]=1.0
        corr = corr / denom
        return corr if max_lag is None else corr[:max_lag]
    def autocorr_diff(p09,p10,max_lag=300):
        p09_ac = np.array([autocorr(x, max_lag=max_lag) for x in p09])
        p10_ac = np.array([autocorr(x, max_lag=max_lag) for x in p10])
        return float(np.mean(np.abs(np.mean(p09_ac,0) - np.mean(p10_ac,0))))
    res['ACD'] = autocorr_diff(R, S, max_lag=min(500, R.shape[1]//2))
    res['SD'] = float(abs(np.mean(skew(R, axis=1)) - np.mean(skew(S, axis=1))))
    res['KD'] = float(abs(np.mean(kurtosis(R, axis=1)) - np.mean(kurtosis(S, axis=1))))
    # CFID
    featsR = extract_psd_feats(R)
    featsS = extract_psd_feats(S)
    if featsR.shape[0] >= 2 and featsS.shape[0] >= 2:
        res['CFID'] = compute_fid_from_feats(featsR, featsS)
    else:
        res['CFID'] = float('nan')
    # DTW
    res['DTW'] = avg_dtw_between_sets(R, S, n_pairs=100)
    return res

def main():
    # user real files
    real_af = Path('PULSOVITAL/npy_output_p09_consolidated/1024seq_AF.npy')
    real_nsr = Path('PULSOVITAL/npy_output_p09_consolidated/1024seq_NSR.npy')
    demos_dir = Path('SSSD-ECG/SSSD-ECG-main/clinical evaluation/diagnosis on normal samples/data')
    demo_files = [demos_dir / 'sample_sssd_norm.npy', demos_dir / 'sample_wavegan_norm.npy', demos_dir / 'sample_p2p_norm.npy', demos_dir / 'sample_real_norm.npy']

    R_af = prepare(np.load(real_af))
    R_nsr = prepare(np.load(real_nsr))

    # choose target length = min among datasets
    lens = [R_af.shape[1], R_nsr.shape[1]] + [np.load(p).shape[1] for p in demo_files]
    target_len = int(min(lens))

    R_af_rs = resample_to(R_af, target_len)
    R_nsr_rs = resample_to(R_nsr, target_len)

    rows = []
    for demo in demo_files:
        S = prepare(np.load(demo))
        S_rs = resample_to(S, target_len)
        # compare AF
        af_res = compute_all_metrics(R_af_rs, S_rs, name=f'AF_vs_{demo.name}')
        af_res.update({'class':'AF','demo':demo.name})
        rows.append(af_res)
        # compare NSR
        nsr_res = compute_all_metrics(R_nsr_rs, S_rs, name=f'NSR_vs_{demo.name}')
        nsr_res.update({'class':'NSR','demo':demo.name})
        rows.append(nsr_res)

    df = pd.DataFrame(rows)
    csvp = Path(OUT_DIR) / 'metrics_master_table_demo_vs_pulso.csv'
    htmlp = Path(OUT_DIR) / 'metrics_master_table_demo_vs_pulso.html'
    df.to_csv(csvp, index=False)
    try:
        styled = df.style.format(na_rep='-', formatter="{:.4f}").render()
        with open(htmlp, 'w', encoding='utf-8') as fh:
            fh.write('<meta charset="utf-8">\n')
            fh.write('<h2>Demo vs PULSOVITAL Metrics</h2>\n')
            fh.write(styled)
    except Exception:
        with open(htmlp, 'w', encoding='utf-8') as fh:
            fh.write(df.to_html(index=False))

    print('Saved CSV:', csvp)
    print('Saved HTML:', htmlp)

if __name__ == '__main__':
    main()
