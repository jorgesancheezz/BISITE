#!/usr/bin/env python3
"""Training wrapper for XResNet1d50 on PTB-XL (paper replication).

This script expects preprocessed numpy arrays (signals and multi-labels).
It will try to use `timm` to create an `xresnet50` backbone with 1D input.

Usage (example):
  python scripts/train_xresnet_wrapper.py --data-dir data/ptbxl_numpy --epochs 50 --batch 32

Important: To truly replicate the paper you should use the exact training
hyperparameters and XResNet1d50 architecture from the paper; this is a wrapper
that attempts to be compatible when `timm` contains xresnet.
"""
import argparse
from pathlib import Path
import sys


def ensure_package(pkg):
    try:
        __import__(pkg)
    except Exception:
        print(f'Install required package: {pkg}\n  pip install {pkg}')
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/ptbxl_numpy')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', default='results/xresnet_ckpt.pth')
    args = p.parse_args()

    ensure_package('torch')
    ensure_package('timm')
    ensure_package('numpy')

    import torch
    import timm
    import numpy as np
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print('Data directory not found:', data_dir)
        print('Run scripts/prepare_ptbxl.py first to create/prep datasets or create numpy arrays with signals and labels.')
        sys.exit(1)

    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')

    # convert to torch tensors (expect shape N, L)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    va = DataLoader(val_ds, batch_size=args.batch)

    # create model via timm if possible
    try:
        model = timm.create_model('xresnet50', pretrained=False, in_chans=1, num_classes=y_train.shape[1])
    except Exception:
        print('timm xresnet50 not available or failed to create; please install a compatible timm version or supply model code.')
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = 0.0
    for ep in range(1, args.epochs+1):
        model.train()
        for xb, yb in tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

        # simple val metric: macro AUROC requires sklearn, we compute val loss as proxy
        model.eval()
        tot_loss = 0.0
        nb = 0
        with torch.no_grad():
            for xb, yb in va:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = loss_fn(out, yb)
                tot_loss += float(loss.item()) * xb.size(0)
                nb += xb.size(0)
        val_loss = tot_loss / max(1, nb)
        print(f'Epoch {ep}/{args.epochs} val_loss={val_loss:.4f}')
        # save checkpoint
        torch.save({'epoch': ep, 'model_state': model.state_dict(), 'val_loss': val_loss}, args.out)


if __name__ == '__main__':
    main()
