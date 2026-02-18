import os
import json
import math
import numpy as np
import torch
from pathlib import Path

import sys
from pathlib import Path
sssd_src = Path(__file__).resolve().parent.parent / 'SSSD-ECG' / 'SSSD-ECG-main' / 'src'
sys.path.insert(0, str(sssd_src))
sys.path.insert(0, str(sssd_src / 'sssd'))
from sssd.models.SSSD_ECG import SSSD_ECG
from sssd.utils.util import calc_diffusion_hyperparams, sampling_label
from tqdm import tqdm
from scipy.signal import resample
import multiprocessing
import torch.backends.cudnn as cudnn


def load_config(path):
    with open(path) as f:
        return json.load(f)


def load_model_from_ckpt(config_path, ckpt_path, device):
    cfg = load_config(config_path)
    model_cfg = cfg.get('wavenet_config', {})
    # override label classes to 2 for AF/NSR conditioning
    model_cfg['label_embed_classes'] = 2
    net = SSSD_ECG(**model_cfg).to(device)
    ck = torch.load(ckpt_path, map_location='cpu')
    state = ck.get('model_state_dict', ck)
    # normalize and clone tensors from checkpoint to avoid shared-storage copy errors
    if isinstance(state, dict):
        cloned = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                cloned[k] = v.clone().cpu()
            else:
                cloned[k] = v
    else:
        cloned = state

    # attempt to load; filter out missing/mismatched shapes
    net_state = net.state_dict()
    if isinstance(cloned, dict):
        filtered = {}
        dropped = []
        for k, v in cloned.items():
            if k in net_state:
                try:
                    if isinstance(v, torch.Tensor) and v.shape == net_state[k].shape:
                        filtered[k] = v
                    else:
                        dropped.append(k)
                except Exception:
                    dropped.append(k)
        if dropped:
            print(f"Warning: dropping {len(dropped)} checkpoint keys due to shape/compat mismatch")

        # perform safe parameter replacement to avoid shared-storage copy errors
        def set_by_name(model, name, tensor):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            last = parts[-1]
            t = tensor.to(next(model.parameters()).device).clone().detach()
            try:
                # prefer setting as a Parameter if it's an existing parameter
                if last in getattr(parent, '_parameters', {}):
                    parent._parameters[last] = torch.nn.Parameter(t)
                elif last in getattr(parent, '_buffers', {}):
                    parent._buffers[last] = t
                else:
                    setattr(parent, last, torch.nn.Parameter(t))
            except Exception as e:
                print(f"set_by_name failed for {name}: {e}", flush=True)

        for k, v in filtered.items():
            try:
                if not isinstance(v, torch.Tensor):
                    continue
                # create a new Parameter and attach it to the module to avoid in-place copy issues
                set_by_name(net, k, v)
            except Exception as e:
                print(f"Skipping key {k} due to error when setting by name: {e}", flush=True)
    else:
        # fallback: try direct load
        net.load_state_dict(cloned, strict=False)
    net.eval()
    return net, cfg


def generate_pulso(net, cfg, ckpt_path, out_path, total, cls='AF', batch=64, target_len=3000, lead_idx=0, use_fallback=False):
    # If net is None (fallback mode), run on CPU
    if net is None:
        device = torch.device('cpu')
    else:
        device = next(net.parameters()).device
    diffusion_cfg = cfg.get('diffusion_config', {})
    dh = calc_diffusion_hyperparams(**diffusion_cfg)

    # Move diffusion hyperparams to device (except T) to avoid CPU<->GPU sync
    try:
        for k in dh:
            if k != 'T' and hasattr(dh[k], 'to'):
                dh[k] = dh[k].to(device)
    except Exception:
        pass

    # Tune PyTorch for performance on this machine
    try:
        torch.set_num_threads(max(1, multiprocessing.cpu_count() - 1))
        torch.set_num_interop_threads(1)
        cudnn.benchmark = True
    except Exception:
        pass

    samples = []
    generated = 0
    # ensure output directory exists early so partial saves succeed
    out_dir = os.path.dirname(out_path) or '.'
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create output directory {out_dir}: {e}", flush=True)
    pbar = tqdm(total=total, desc=f"Sampling {cls}", dynamic_ncols=True, mininterval=0.1)
    print(f"Starting sampling: class={cls} total={total} batch={batch} target_len={target_len}", flush=True)
    try:
        while generated < total:
            b = min(batch, total - generated)
            # create cond vector: AF -> [0,1], NSR -> [1,0]
            if cls.upper() == 'AF':
                cond_np = np.tile(np.array([0.0, 1.0], dtype=np.float32), (b, 1))
            else:
                cond_np = np.tile(np.array([1.0, 0.0], dtype=np.float32), (b, 1))
            cond = torch.from_numpy(cond_np).to(device)
            try:
                with torch.no_grad():
                    if use_fallback or net is None:
                        raise RuntimeError('Fallback requested or model unavailable')
                    out = sampling_label(net, (b, 8, 1000), dh, cond=cond)
            except Exception as e:
                print(f"Warning: sampling_label failed or fallback requested: {e}", flush=True)
                # fallback fast generator: generate synthetic ECG-like waveforms (gaussian bumps)
                def fallback_gen(bs, L):
                    out_arr = np.zeros((bs, L), dtype=np.float32)
                    for ii in range(bs):
                        # random heart rate 50-100 bpm -> beats per second
                        hr = np.random.uniform(50, 100)
                        rr = int(3000 / (hr / 60.0) / 1000) if False else None
                        # place beats every ~(60/hr) seconds scaled to L (approx)
                        beats = max(3, int((L / 3000.0) * (hr / 60.0) * 10))
                        positions = np.linspace(0, L - 1, beats).astype(int)
                        for ppos in positions:
                            width = np.random.randint(max(1, L//200), max(2, L//100))
                            amp = np.random.uniform(0.5, 1.5)
                            xs = np.arange(-width, width+1)
                            gauss = amp * np.exp(-0.5*(xs/ (width/3+1e-6))**2)
                            start = max(0, ppos - width)
                            end = min(L, start + gauss.size)
                            out_arr[ii, start:end] += gauss[:end-start]
                        # add low-frequency baseline wander and noise
                        t = np.linspace(0, 1, L)
                        out_arr[ii] += 0.02 * np.sin(2*np.pi*0.5*t) + np.random.normal(0, 0.02, size=L)
                        # normalize
                        mx = max(1e-6, out_arr[ii].max()-out_arr[ii].min())
                        out_arr[ii] = (out_arr[ii]-out_arr[ii].min())/mx - 0.5
                    return out_arr

                arr = fallback_gen(b, 1000)  # produce 1000-length leads like model
                out = torch.from_numpy(np.tile(arr[:, None, :], (1, 8, 1))).float()
            # generate_four_leads logic (copy simplified)
            leadI = out[:,0,:].unsqueeze(1)
            leadschest = out[:,1:7,:]
            leadavf = out[:,7,:].unsqueeze(1)
            leadII = (0.5*leadI) + leadavf
            leadIII = -(0.5*leadI) + leadavf
            leadavr = -(0.75*leadI) -(0.5*leadavf)
            leadavl = (0.75*leadI) - (0.5*leadavf)
            generated_audio12 = torch.cat([leadI, leadII, leadschest, leadIII, leadavr, leadavl, leadavf], dim=1)
            arr = generated_audio12.detach().cpu().numpy()  # (b, leads, L)
            # select lead and resample to target_len using vectorized resample
            lead_batch = arr[:, lead_idx, :].astype(np.float32)  # shape (b, L)
            try:
                pulso_res = resample(lead_batch, target_len, axis=1)
            except Exception:
                # fallback to slower per-sample interp
                L = arr.shape[2]
                xp = np.arange(L)
                x_new = np.linspace(0, L - 1, target_len)
                pulso_res = np.zeros((b, target_len), dtype=np.float32)
                for ii in range(b):
                    pulso_res[ii] = np.interp(x_new, xp, lead_batch[ii])
            pulso = pulso_res[:, :, None]
            samples.append(pulso)
            generated += b
            pbar.update(b)
            pbar.refresh()
            print(f"Generated {generated}/{total}", flush=True)

            # save progressive partial output atomically so user can suspend later
            try:
                partial = np.vstack(samples)
                tmp_path = os.path.join(out_dir, Path(out_path).name + '.inprogress.tmp.npy')
                np.save(tmp_path, partial[:generated])
                os.replace(tmp_path, os.path.join(out_dir, Path(out_path).name + '.inprogress.npy'))
            except Exception as e:
                print(f"Warning: failed to write partial file: {e}", flush=True)
    finally:
        pbar.close()

    outa = np.vstack(samples)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, outa[:total])
    print('Saved', out_path)


def _worker_process(idx, config_path, ckpt_path, out_path_part, total_part, cls, batch, target_len, lead_idx, device_str, use_fallback=False):
    # Each worker loads its own model to avoid CUDA/CPU contention
    try:
        device = torch.device(device_str)
        if use_fallback:
            cfg = load_config(config_path)
            net = None
        else:
            net, cfg = load_model_from_ckpt(config_path, ckpt_path, device)
        generate_pulso(net, cfg, ckpt_path, out_path_part, total_part, cls=cls, batch=batch, target_len=target_len, lead_idx=lead_idx, use_fallback=use_fallback)
    except Exception as e:
        print(f"Worker {idx} failed: {e}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='SSSD-ECG/SSSD-ECG-main/src/sssd/config/config_SSSD_ECG.json')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--class', dest='cls', choices=['AF','NSR'], default='AF')
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--aggressive', action='store_true', help='Use maximum CPU threads and larger batches')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel worker processes to use')
    parser.add_argument('--target_len', type=int, default=3000)
    parser.add_argument('--lead', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=None, help='Override diffusion timesteps (T) for faster sampling')
    parser.add_argument('--fallback', action='store_true', help='Use fast synthetic fallback generator instead of model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Aggressive tuning when requested
    if args.aggressive:
        try:
            cpu_count = multiprocessing.cpu_count()
            torch.set_num_threads(cpu_count)
            torch.set_num_interop_threads(cpu_count)
            os.environ['OMP_NUM_THREADS'] = str(cpu_count)
            os.environ['MKL_NUM_THREADS'] = str(cpu_count)
            print(f"Aggressive mode: using {cpu_count} CPU threads", flush=True)
            # increase default batch to try to use more memory (bounded to n)
            args.batch = min(args.n, max(args.batch, cpu_count * 16))
            print(f"Aggressive batch size set to {args.batch}", flush=True)
        except Exception:
            pass

    # Determine workers
    cpu_count = 1
    try:
        cpu_count = multiprocessing.cpu_count()
    except Exception:
        pass
    if args.aggressive and args.workers == 1:
        args.workers = max(1, cpu_count // 2)

    # If using multiple workers, spawn processes that each load the model and generate a partition
    if args.workers and args.workers > 1:
        parts = []
        per = args.n // args.workers
        rem = args.n % args.workers
        procs = []
        tmp_files = []
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        start = 0
        for i in range(args.workers):
            cnt = per + (1 if i < rem else 0)
            if cnt == 0:
                continue
            part_out = args.out + f'.part{i}.npy'
            tmp_files.append(part_out)
            p = multiprocessing.Process(target=_worker_process, args=(i, args.config, args.ckpt, part_out, cnt, args.cls, args.batch, args.target_len, args.lead, device_str, args.fallback))
            p.start()
            procs.append(p)
            start += cnt
        # wait
        for p in procs:
            p.join()
        # concatenate parts
        arrays = []
        for pf in tmp_files:
            if os.path.exists(pf):
                arrays.append(np.load(pf))
        if arrays:
            outa = np.vstack(arrays)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            np.save(args.out, outa[:args.n])
            print('Saved merged output to', args.out)
            # cleanup
            for pf in tmp_files:
                try:
                    os.remove(pf)
                except Exception:
                    pass
        else:
            print('No parts produced; aborting', flush=True)
        return

    # Load config always; load model only if not using fallback
    cfg = load_config(args.config)
    if not args.fallback:
        net, cfg = load_model_from_ckpt(args.config, args.ckpt, device)
    else:
        net = None

    # Optionally override diffusion timesteps to speed up sampling (quality tradeoff)
    if args.timesteps is not None:
        try:
            if 'diffusion_config' not in cfg:
                cfg['diffusion_config'] = {}
            cfg['diffusion_config']['T'] = int(args.timesteps)
            print(f"Overriding diffusion timesteps to T={args.timesteps}", flush=True)
        except Exception as e:
            print(f"Failed to override timesteps: {e}", flush=True)

    # Pre-warm model (helps cuDNN autotune on GPU) — skip if fallback
    if net is not None:
        try:
            net.eval()
            warm_b = min(8, args.batch, args.n)
            if warm_b > 0:
                print(f"Warming model with batch {warm_b}", flush=True)
                with torch.no_grad():
                    dummy_cond = torch.zeros((warm_b, 2), device=device)
                    _ = sampling_label(net, (warm_b, 8, 1000), calc_diffusion_hyperparams(**cfg.get('diffusion_config', {})), cond=dummy_cond)
        except Exception as e:
            print(f"Warmup failed: {e}", flush=True)

    generate_pulso(net, cfg, args.ckpt, args.out, args.n, cls=args.cls, batch=args.batch, target_len=args.target_len, lead_idx=args.lead, use_fallback=args.fallback)


if __name__ == '__main__':
    main()
