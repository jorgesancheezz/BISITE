#!/usr/bin/env python3
"""Compute alternate metrics table (tSNE, DTW, mmd2, mmd_sigma, energy, psd_12, discriminator_auc)
for the AF and NSR pairs and save CSV/HTML in `notebooks/outputs/`.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from tools.compare_with_article_demos import (
    prepare, resample_to, median_rbf_sigma, rbf_mmd2,
    energy_distance_flat, extract_psd_feats, discriminator_auc, avg_dtw_between_sets
)

PAIRS = [
    (Path('PULSOVITAL/Metricas/1024seq_AF.npy'), Path('PULSOVITAL/Metricas/sssd_article_AF_proc_3000.npy'), 'AF'),
    (Path('PULSOVITAL/Metricas/1024seq_NSR.npy'), Path('PULSOVITAL/Metricas/sssd_article_NSR_proc_3000.npy'), 'NSR'),
]


def compute_tsne_dist(A, B, n_pca=50):
    # A,B: (N,T)
    X = np.vstack([A, B])
    n = len(A)
    # PCA to reduce dims
    k = min(n_pca, X.shape[1])
    if k <= 0:
        return float('nan')
    pca = PCA(n_components=k)
    Xp = pca.fit_transform(X)
    # t-SNE to 2D (small perplexity to be stable on small N)
    try:
        ts = TSNE(n_components=2, perplexity=min(30, max(5, int(len(X)/3))), random_state=42, init='pca')
        Z = ts.fit_transform(Xp)
        Za = Z[:n]; Zb = Z[n:]
        ca = Za.mean(axis=0); cb = Zb.mean(axis=0)
        dist = float(np.linalg.norm(ca - cb))
        return dist
    except Exception:
        return float('nan')


def compute_pca_centroid_dist(A, B, n_pca=50):
    # stable distance: Euclidean between PCA centroids
    X = np.vstack([A, B])
    k = min(n_pca, X.shape[1])
    if k <= 0:
        return float('nan')
    pca = PCA(n_components=k)
    Xp = pca.fit_transform(X)
    n = len(A)
    Za = Xp[:n]; Zb = Xp[n:]
    ca = Za.mean(axis=0); cb = Zb.mean(axis=0)
    return float(np.linalg.norm(ca - cb))


def main():
    out_dir = Path('notebooks/outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for a_path, b_path, cls in PAIRS:
        print('Processing', a_path, 'vs', b_path)
        if not a_path.exists() or not b_path.exists():
            print('Missing files for', cls); continue
        A = prepare(np.load(a_path))
        B = prepare(np.load(b_path))
        # align lengths
        target_len = int(min(A.shape[1], B.shape[1]))
        A_rs = resample_to(A, target_len)
        B_rs = resample_to(B, target_len)

        # tSNE centroid distance
        tsne_dist = compute_tsne_dist(A_rs, B_rs, n_pca=50)

        # DTW
        dtw = avg_dtw_between_sets(A_rs, B_rs, n_pairs=100)

        # MMD
        sigma = median_rbf_sigma(A_rs, B_rs)
        mmd2 = rbf_mmd2(A_rs, B_rs, sigma)

        # energy
        energy = energy_distance_flat(A_rs, B_rs)

        # psd_12: L2 between mean PSD PCA features (n_comp=12)
        featsA = extract_psd_feats(A_rs, n_comp=12)
        featsB = extract_psd_feats(B_rs, n_comp=12)
        if featsA.size and featsB.size:
            # compute mean feature vectors and L2
            meanA = featsA.mean(axis=0)
            meanB = featsB.mean(axis=0)
            psd_12 = float(np.linalg.norm(meanA - meanB))
        else:
            psd_12 = float('nan')

        # discriminator_auc
        disc_auc = discriminator_auc(A_rs, B_rs)

        pca_cent = compute_pca_centroid_dist(A_rs, B_rs, n_pca=50)
        rows.append({'CLASS': cls,
                 'tSNE': tsne_dist,
                 'PCA_centroid': pca_cent,
                 'DTW': dtw,
                 'mmd2': mmd2,
                 'mmd_sigma': sigma,
                 'energy': energy,
                 'psd_12': psd_12,
                 'discriminator_auc': disc_auc})

    df = pd.DataFrame(rows)
    csvp = out_dir / 'pair_metrics_variant.csv'
    htmlp = out_dir / 'pair_metrics_variant.html'
    # format numeric columns to 3 decimals for CSV/HTML
    numcols = df.select_dtypes(include=[float]).columns.tolist()
    df[numcols] = df[numcols].round(3)
    df.to_csv(csvp, index=False, float_format='%.3f')
    try:
        styled = df.style.format(na_rep='-').render()
        with open(htmlp, 'w', encoding='utf-8') as fh:
            fh.write('<meta charset="utf-8">\n')
            fh.write('<h2>Pair metrics variant</h2>\n')
            fh.write(styled)
    except Exception:
        with open(htmlp, 'w', encoding='utf-8') as fh:
            fh.write(df.to_html(index=False))

    print('Saved CSV:', csvp)
    print('Saved HTML:', htmlp)
    print(df.to_string(index=False, float_format='{:0.4f}'.format))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
