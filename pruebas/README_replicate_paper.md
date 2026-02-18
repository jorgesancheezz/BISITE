# Reproduce paper experiment (SSSD-ECG / XResNet1d50)

This workspace contains helper scripts to reproduce the paper's evaluation pipeline at a high level.

Steps to reproduce (recommended):

1) Prepare environment

```
python -m venv .venv-ptbxl
.venv-ptbxl\Scripts\activate
pip install -U pip
pip install numpy scipy matplotlib scikit-learn wfdb ptbxl timm torch
```

2) Download / prepare PTB-XL

Use `scripts/prepare_ptbxl.py` to download metadata and prepare numpy arrays. The script currently contains guidance; depending on network you may need to download signals via PhysioNet WFDB tools.

```
python scripts/prepare_ptbxl.py --out data/ptbxl --download
```

3) Preprocess to numpy arrays

Create `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy` under `data/ptbxl_numpy`. Each `X_*` should be shape `(N, L)` and `y_*` shape `(N, 71)` (multi-label binary).

4) Train XResNet1d50

```
python scripts/train_xresnet_wrapper.py --data-dir data/ptbxl_numpy --epochs 50 --batch 32 --out results/xresnet_ckpt.pth
```

Notes and caveats:
- The exact replication requires the XResNet1d50 architecture and training hyperparameters used in the paper. The wrapper attempts to use `timm`'s `xresnet50` model; if behavior differs, adapt the model code.
- Training PTB-XL is compute-heavy; use GPU and expect many hours depending on hardware.
- After obtaining a trained checkpoint, evaluate synthetic datasets by converting your synthetic samples to the same format and running inference producing per-label probabilities to compute macro-AUROC.

If you want, I can:
- attempt to download PTB-XL and prepare arrays here (will use network and take time),
- or generate the preprocessing code to convert PTB-XL WFDB records to numpy arrays and the evaluation script that computes macro-AUROC per the paper.
