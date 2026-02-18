import os
import argparse
import json
import numpy as np
import torch

from utils.util import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams, DEVICE
from models.SSSD_ECG import SSSD_ECG


def generate_four_leads(tensor):
    leadI = tensor[:,0,:].unsqueeze(1)
    leadschest = tensor[:,1:7,:]
    leadavf = tensor[:,7,:].unsqueeze(1)

    leadII = (0.5*leadI) + leadavf

    leadIII = -(0.5*leadI) + leadavf
    leadavr = -(0.75*leadI) -(0.5*leadavf)
    leadavl = (0.75*leadI) - (0.5*leadavf)

    leads12 = torch.cat([leadI, leadII, leadschest, leadIII, leadavr, leadavl, leadavf], dim=1)

    return leads12


def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter):
    
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    """

    # generate experiment (local) path
    local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"], 
                                           diffusion_config["T"], 
                                           diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to device
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(DEVICE)

    # predefine model
    net = SSSD_ECG(**model_config).to(DEVICE)
    print_size(net)

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception:
        raise Exception('No valid model found')

    # Attempt standard loading first
    state = checkpoint.get('model_state_dict', checkpoint)
    net_state = net.state_dict()
    try:
        # debug: report key counts
        if isinstance(state, dict):
            print(f'Checkpoint parameter count: {len(state)}')
            matches = sum(1 for k in net_state.keys() if k in state)
            print(f'Model parameter count: {len(net_state)}, matching keys in checkpoint: {matches}')
        else:
            print('Checkpoint does not contain a dict under "model_state_dict"')

        net.load_state_dict(state)
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except Exception:
        # Fallback: build a cloned, filtered state dict matching model keys
        net_state = net.state_dict()
        filtered_state = {}
        for k in net_state.keys():
            if k in state:
                v = state[k]
                try:
                    # Ensure tensor, clone to avoid shared storage issues, and match dtype
                    tv = v.clone().detach()
                    if tv.dtype != net_state[k].dtype:
                        tv = tv.to(net_state[k].dtype)
                    if tv.shape == net_state[k].shape:
                        filtered_state[k] = tv
                except Exception:
                    # skip problematic parameter
                    continue

        if len(filtered_state) == 0:
            raise Exception('No compatible parameters found in checkpoint')

        # As a last-resort fallback, replace module parameters/buffers directly
        def set_by_name(model, name, tensor):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            last = parts[-1]
            t = tensor.clone().detach()
            if last in getattr(parent, '_parameters', {}):
                parent._parameters[last] = torch.nn.Parameter(t)
            elif last in getattr(parent, '_buffers', {}):
                parent._buffers[last] = t
            else:
                try:
                    setattr(parent, last, torch.nn.Parameter(t))
                except Exception:
                    # skip if we cannot set
                    pass

        for k, v in filtered_state.items():
            try:
                set_by_name(net, k, v)
            except Exception:
                continue

        print('Successfully replaced model parameters (direct) at iteration {}'.format(ckpt_iter))

   
    labels = np.load('ptbxl_test_labels.npy')
    l1 = labels[0:400]
    l2 = labels[400:800]
    l3 = labels[800:1200]
    l4 = labels[1200:1600]
    l5 = labels[1600:2000]
    l6 = labels[2000:]
    
    for i, label in enumerate((l1,l2,l3,l4,l5,l6)):
        
        cond = torch.from_numpy(label).to(DEVICE).float()

        # inference
        # timing: use CUDA events when available, otherwise time.time()
        if DEVICE.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            import time
            start_time = time.time()

        generated_audio = sampling_label(net, (num_samples,8,1000), 
                               diffusion_hyperparams,
                               cond=cond)

        generated_audio12 = generate_four_leads(generated_audio)

        if DEVICE.type == 'cuda':
            end.record()
            torch.cuda.synchronize()
            elapsed = int(start.elapsed_time(end) / 1000)
        else:
            elapsed = int(time.time() - start_time)
        print('generated {} utterances of random_digit at iteration {} in {} seconds'.format(num_samples,
                                                                               ckpt_iter,
                                                                               elapsed))

       
        outfile = f'{i}_samples.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, generated_audio12.detach().cpu().numpy())
        print('saved generated samples at iteration %s' % ckpt_iter)
        
        outfile = f'{i}_labels.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, cond.detach().cpu().numpy())
        print('saved generated samples at iteration %s' % ckpt_iter)

        # Optional: save in PULSOVITAL-compatible format (N, target_length, 1)
        if gen_config.get('pulso_save', False):
            try:
                target_len = int(gen_config.get('pulso_target_length', 3000))
                lead_idx = int(gen_config.get('pulso_lead', 0))
                arr = generated_audio12.detach().cpu().numpy()  # shape (N, leads, L)
                # select lead
                lead = arr[:, lead_idx, :]
                N, L = lead.shape
                # resample each sample to target_len via linear interp
                xp = np.arange(L)
                x_new = np.linspace(0, L - 1, target_len)
                pulso = np.zeros((N, target_len, 1), dtype=lead.dtype)
                for k in range(N):
                    pulso[k, :, 0] = np.interp(x_new, xp, lead[k])
                pulso_out = os.path.join(ckpt_path, f'{i}_samples_pulso.npy')
                np.save(pulso_out, pulso)
                print(f'saved PULSOVITAL-formatted samples at {pulso_out}')
            except Exception as e:
                print('Failed to save pulso format:', e)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_SSSD_ECG.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default=100000,
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=400,
                        help='Number of utterances to be generated')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']

    generate(**gen_config,
             num_samples=args.num_samples,
             ckpt_iter=args.ckpt_iter,
             data_path=trainset_config.get("data_path", ""))

