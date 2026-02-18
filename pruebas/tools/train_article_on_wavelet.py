import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
import sys

# ensure repo src in path
sssd_src = Path(__file__).resolve().parent.parent / 'SSSD-ECG' / 'SSSD-ECG-main' / 'src'
sys.path.insert(0, str(sssd_src))
sys.path.insert(0, str(sssd_src / 'sssd'))
# Provide a lightweight shim for pytorch_lightning.utilities.rank_zero_only if missing
try:
    import pytorch_lightning.utilities as _pl_utils  # type: ignore
except Exception:
    import types
    pl_utils = types.ModuleType('pytorch_lightning.utilities')
    def rank_zero_only(fn):
        return fn
    pl_utils.rank_zero_only = rank_zero_only
    sys.modules['pytorch_lightning.utilities'] = pl_utils
    sys.modules['pytorch_lightning'] = types.ModuleType('pytorch_lightning')

# import sssd utilities
from sssd.utils.util import calc_diffusion_hyperparams, training_loss_label, print_size

from sssd.models.SSSD_ECG import SSSD_ECG

# import WaveletDataGenerator from PULSOVITAL
try:
    from PULSOVITAL.training.train import WaveletDataGenerator
except Exception:
    try:
        import importlib.util, sys
        mod_path = str(Path(__file__).resolve().parent.parent / 'PULSOVITAL' / 'training' / 'train.py')
        spec = importlib.util.spec_from_file_location('pulso_train_module', mod_path)
        pulso_mod = importlib.util.module_from_spec(spec)
        sys.modules['pulso_train_module'] = pulso_mod
        spec.loader.exec_module(pulso_mod)
        WaveletDataGenerator = getattr(pulso_mod, 'WaveletDataGenerator')
    except Exception as e:
        WaveletDataGenerator = None


def make_batch(gen, n_af, n_nsr, length, alpha_af, alpha_nsr, device):
    import torch as _torch
    Xs = []
    Ys = []
    if n_af > 0:
        for s in gen(n_af, alpha_af):
            arr = s.cpu().numpy() if hasattr(s, 'cpu') else np.asarray(s)
            Xs.append(arr.reshape(-1)[:length])
            Ys.append([0,1])
    if n_nsr > 0:
        for s in gen(n_nsr, alpha_nsr):
            arr = s.cpu().numpy() if hasattr(s, 'cpu') else np.asarray(s)
            Xs.append(arr.reshape(-1)[:length])
            Ys.append([1,0])
    Xs = np.stack(Xs, axis=0).astype(np.float32)
    # shape to (B,1,L)
    Xs = _torch.from_numpy(Xs).unsqueeze(1).to(device)
    Ys = _torch.tensor(Ys, dtype=_torch.float32).to(device)
    # shuffle
    idx = _torch.randperm(Xs.shape[0])
    return Xs[idx], Ys[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sssd_config', type=str, default='SSSD-ECG/SSSD-ECG-main/config/SSSD_ECG.json')
    parser.add_argument('--out', type=str, default='SSSD_trained_on_wavelet')
    parser.add_argument('--n_iters', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ckpt_every', type=int, default=500)
    parser.add_argument('--alpha_af', type=float, default=0.8)
    parser.add_argument('--alpha_nsr', type=float, default=0.2)
    parser.add_argument('--length', type=int, default=3000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--small_test', action='store_true')
    args = parser.parse_args()

    if WaveletDataGenerator is None:
        raise RuntimeError('WaveletDataGenerator not available; cannot proceed')

    with open(args.sssd_config) as f:
        config = json.load(f)

    model_config = config.get('wavenet_config', {})
    diffusion_config = config.get('diffusion_config', {})

    # override channels and conditioning for two-class setup
    model_config['in_channels'] = 1
    model_config['out_channels'] = 1
    model_config['label_embed_classes'] = 2

    device = torch.device(args.device)

    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    net = SSSD_ECG(**model_config).to(device)
    print_size(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    gen = WaveletDataGenerator(length=args.length, noise_scale=0.0)

    n_iter = 0
    while n_iter < args.n_iters:
        n_iter += 1
        n_af = args.batch_size // 2
        n_nsr = args.batch_size - n_af
        X, Y = make_batch(gen, n_af, n_nsr, args.length, args.alpha_af, args.alpha_nsr, device)

        optimizer.zero_grad()
        loss = training_loss_label(net, nn.MSELoss(), (X, Y), diffusion_hyperparams)
        loss.backward()
        optimizer.step()

        if n_iter % args.save_every == 0:
            print(f'iter {n_iter} loss {loss.item():.6f}')

        if n_iter % args.ckpt_every == 0:
            os.makedirs(args.out, exist_ok=True)
            torch.save({'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.out, f'{n_iter}.pkl'))
            print('Saved checkpoint', n_iter)


if __name__ == '__main__':
    main()
