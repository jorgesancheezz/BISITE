import argparse
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import zscore


def project_first_pc(X):
    X = np.asarray(X)
    # X: (n_samples, time)
    Xc = X - X.mean(axis=1, keepdims=True)
    try:
        u,s,vt = np.linalg.svd(Xc, full_matrices=False)
        pc = vt[0]
        proj = Xc.dot(pc)
        return proj
    except Exception:
        return X.mean(axis=1)


def normalize_per_signal(A):
    # z-score per signal (sample)
    return zscore(A, axis=1, ddof=0)


def load_and_prep(path, normalize=True):
    A = np.load(path)
    if A.ndim==3:
        A = A.reshape(A.shape[0], -1)
    elif A.ndim==1:
        A = A.reshape(1, -1)
    elif A.ndim==2 and A.shape[1]==1:
        A = A.reshape(A.shape[0], -1)
    if normalize:
        try:
            A = normalize_per_signal(A)
        except Exception:
            pass
    return A


def make_report(real_path, synth_path, outdir, seed=42, normalize=True, n_examples=3):
    os.makedirs(outdir, exist_ok=True)
    R = load_and_prep(real_path, normalize=normalize)
    S = load_and_prep(synth_path, normalize=normalize)
    rng = np.random.default_rng(seed)

    # projections
    Rproj = project_first_pc(R)
    Sproj = project_first_pc(S)

    # histogram plot
    plt.figure(figsize=(8,5))
    bins = 80
    plt.hist(Rproj, bins=bins, density=True, alpha=0.6, label='Real')
    plt.hist(Sproj, bins=bins, density=True, alpha=0.6, label='Synth')
    plt.legend()
    plt.title('1D projection histogram (first PC)')
    plt.xlabel('Projection value')
    plt.ylabel('Density')
    hist_file = os.path.join(outdir, 'projection_hist.png')
    plt.tight_layout()
    plt.savefig(hist_file, dpi=150)
    plt.close()

    # example traces
    nR = R.shape[0]
    nS = S.shape[0]
    idxR = rng.choice(nR, size=min(n_examples, nR), replace=False)
    idxS = rng.choice(nS, size=min(n_examples, nS), replace=False)

    fig, axs = plt.subplots(n_examples, 2, figsize=(10, 2.5*n_examples), sharex=False)
    for i,ir in enumerate(idxR):
        axs[i,0].plot(R[ir], color='C0')
        axs[i,0].set_title(f'Real sample {ir}')
        axs[i,0].set_ylabel('Amplitude')
    for i,is_ in enumerate(idxS):
        axs[i,1].plot(S[is_], color='C1')
        axs[i,1].set_title(f'Synth sample {is_}')
    plt.tight_layout()
    traces_file = os.path.join(outdir, 'example_traces.png')
    plt.savefig(traces_file, dpi=150)
    plt.close()

    # small numeric summary file
    summary = {
        'real_n': int(R.shape[0]),
        'synth_n': int(S.shape[0]),
        'real_time_len': int(R.shape[1]),
        'synth_time_len': int(S.shape[1]),
        'real_proj_mean': float(np.mean(Rproj)),
        'synth_proj_mean': float(np.mean(Sproj)),
        'real_proj_std': float(np.std(Rproj)),
        'synth_proj_std': float(np.std(Sproj))
    }
    import json
    with open(os.path.join(outdir,'report_summary.json'),'w',encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    return hist_file, traces_file, os.path.join(outdir,'report_summary.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', required=True)
    parser.add_argument('--synth', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    args = parser.parse_args()
    hi, tr, js = make_report(args.real, args.synth, args.out, seed=args.seed, normalize=args.normalize)
    print('WROTE', hi)
    print('WROTE', tr)
    print('WROTE', js)

if __name__=='__main__':
    main()
