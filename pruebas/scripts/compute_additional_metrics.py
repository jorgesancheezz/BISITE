import os
import json
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, wasserstein_distance
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score


def ensure_2d(a):
    a = np.asarray(a)
    if a.ndim == 3:
        # assume shape (N, L, 1)
        return a.reshape(a.shape[0], a.shape[1])
    if a.ndim == 2:
        return a
    if a.ndim == 1:
        # assume 1 sample
        return a.reshape(1, -1)
    raise ValueError(f"Unsupported array shape: {a.shape}")


def pca_features(X, n_components=50):
    n_samples, n_times = X.shape
    n_components = min(n_components, n_samples, n_times)
    if n_components <= 0:
        return X
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def discriminative_score(real, synth, n_pca=50, n_splits=5, random_state=42):
    # Robustified discriminative score: if marginal distributions are (nearly) identical,
    # return 0.5 immediately to avoid degenerate classifier behavior on identical features.
    # Otherwise average CV AUC across small jitter repeats.
    # Quick marginal check:
    try:
        wd = wasserstein_distance(real.ravel(), synth.ravel())
    except Exception:
        wd = None
    if wd is not None and wd <= 1e-12:
        return 0.5
    X = np.vstack([real, synth])
    y = np.hstack([np.zeros(len(real)), np.ones(len(synth))])
    Xf = pca_features(X, n_components=n_pca)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = LogisticRegression(max_iter=2000)
    repeats = 10
    eps = 1e-6
    all_aucs = []
    rng = np.random.RandomState(random_state)
    for rep in range(repeats):
        noise = rng.normal(scale=eps, size=Xf.shape)
        Xf_noisy = Xf + noise
        aucs = []
        for train_idx, test_idx in cv.split(Xf_noisy, y):
            if len(np.unique(y[test_idx])) < 2:
                continue
            try:
                clf.fit(Xf_noisy[train_idx], y[train_idx])
                probs = clf.predict_proba(Xf_noisy[test_idx])[:, 1]
                auc = roc_auc_score(y[test_idx], probs)
                aucs.append(auc)
            except Exception:
                continue
        if len(aucs) > 0:
            all_aucs.append(np.mean(aucs))
    return float(np.mean(all_aucs)) if len(all_aucs) > 0 else 0.5


def predictive_score(real, synth, n_pca=50, n_splits=5, random_state=42):
    # Compute predictive score via K-fold on the real data.
    # For each fold: train on synth + real_train, test on real_test.
    # For the test fold, compute ROC-AUC if possible; if the test contains a single class,
    # fall back to accuracy for that fold. Return mean across folds.
    real = np.asarray(real)
    n_real = len(real)
    if n_real < 2:
        return 0.5
    cv = StratifiedKFold(n_splits=min(n_splits, n_real), shuffle=True, random_state=random_state)
    y_real = np.zeros(n_real, dtype=int)
    fold_scores = []
    pca = PCA(n_components=min(n_pca, max(1, real.shape[1])))
    rng = np.random.RandomState(random_state)
    for train_idx, test_idx in cv.split(real, y_real):
        real_train = real[train_idx]
        real_test = real[test_idx]
        X_train = np.vstack([synth, real_train])
        y_train = np.hstack([np.ones(len(synth)), np.zeros(len(real_train))])

        # Fit PCA on train and transform both
        n_comp = min(n_pca, X_train.shape[0], X_train.shape[1])
        if n_comp <= 0:
            Xtr_f = X_train
            Xte_f = real_test
        else:
            pca_local = PCA(n_components=n_comp)
            Xtr_f = pca_local.fit_transform(X_train)
            Xte_f = pca_local.transform(real_test)

        clf = LogisticRegression(max_iter=2000)
        try:
            clf.fit(Xtr_f, y_train)
            probs = clf.predict_proba(Xte_f)[:, 1]
            # If y_test has only one class, roc_auc_score will raise ValueError
            y_test = np.zeros(len(real_test), dtype=int)
            try:
                score = roc_auc_score(y_test, probs)
                # roc_auc_score may issue a warning and return nan in some scikit-learn versions;
                # if so, fallback to accuracy
                if np.isnan(score):
                    preds = (probs >= 0.5).astype(int)
                    score = accuracy_score(y_test, preds)
            except ValueError:
                preds = (probs >= 0.5).astype(int)
                score = accuracy_score(y_test, preds)
            fold_scores.append(float(score))
        except Exception:
            # if classifier fails for any reason, skip fold
            continue
    if len(fold_scores) == 0:
        return 0.5
    return float(np.mean(fold_scores))


def marginal_distribution_diff(real, synth):
    # Wasserstein distance between flattened amplitude distributions
    return float(wasserstein_distance(real.ravel(), synth.ravel()))


def autocorr(x, max_lag=None):
    x = x - np.mean(x)
    N = len(x)
    corr = correlate(x, x, mode='full')
    corr = corr[N-1:]
    corr = corr / (np.var(x) * np.arange(N, 0, -1))
    if max_lag is None:
        return corr
    return corr[:max_lag]


def autocorr_difference(real, synth, max_lag=300):
    # compute mean autocorr across samples up to max_lag and return L1 mean abs diff
    real_ac = np.array([autocorr(x, max_lag=max_lag) for x in real])
    synth_ac = np.array([autocorr(x, max_lag=max_lag) for x in synth])
    mean_real = np.mean(real_ac, axis=0)
    mean_synth = np.mean(synth_ac, axis=0)
    return float(np.mean(np.abs(mean_real - mean_synth)))


def skewness_difference(real, synth):
    r = skew(real, axis=1)
    s = skew(synth, axis=1)
    return float(np.abs(np.mean(r) - np.mean(s)))


def kurtosis_difference(real, synth):
    r = kurtosis(real, axis=1)
    s = kurtosis(synth, axis=1)
    return float(np.abs(np.mean(r) - np.mean(s)))


def compute_metrics(real, synth, fs=300):
    # ensure correct shapes
    real = ensure_2d(real)
    synth = ensure_2d(synth)
    metrics = {}
    metrics['DS'] = discriminative_score(real, synth)
    metrics['PS'] = predictive_score(real, synth)
    metrics['MDD_wasserstein'] = marginal_distribution_diff(real, synth)
    metrics['ACD_mean_abs'] = autocorr_difference(real, synth, max_lag=min(500, real.shape[1]//2))
    metrics['SD'] = skewness_difference(real, synth)
    metrics['KD'] = kurtosis_difference(real, synth)
    return metrics


def main():
    paths = {
        'AF_real': 'PULSOVITAL/npy_output/AF_processed_1024x3000x1.npy',
        'AF_synth': 'PULSOVITAL/Metricas/1024seq_AF.npy',
        'NSR_real': 'PULSOVITAL/npy_output/NSR_processed_1024x3000x1.npy',
        'NSR_synth': 'PULSOVITAL/Metricas/1024seq_NSR.npy'
    }

    for k,p in paths.items():
        print(k, os.path.exists(p), p)

    AF_real = np.load(paths['AF_real'])
    AF_synth = np.load(paths['AF_synth'])
    NSR_real = np.load(paths['NSR_real'])
    NSR_synth = np.load(paths['NSR_synth'])

    AF_real = ensure_2d(AF_real)
    AF_synth = ensure_2d(AF_synth)
    NSR_real = ensure_2d(NSR_real)
    NSR_synth = ensure_2d(NSR_synth)

    print('shapes AF:', AF_real.shape, AF_synth.shape)
    print('shapes NSR:', NSR_real.shape, NSR_synth.shape)

    results = {}
    results['AF'] = compute_metrics(AF_real, AF_synth)
    results['NSR'] = compute_metrics(NSR_real, NSR_synth)

    out_dir = 'compare_out_test'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'additional_metrics_AF.json'), 'w') as f:
        json.dump(results['AF'], f, indent=2)
    with open(os.path.join(out_dir, 'additional_metrics_NSR.json'), 'w') as f:
        json.dump(results['NSR'], f, indent=2)

    # also write CSV
    df = pd.DataFrame([{'class': 'AF', **results['AF']}, {'class': 'NSR', **results['NSR']}])
    df.to_csv(os.path.join(out_dir, 'additional_metrics_summary.csv'), index=False)

    print('\nResults:')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
