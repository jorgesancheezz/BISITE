import os
import json
import argparse
import numpy as np
import importlib.util

# Import compute_metrics from scripts/compute_additional_metrics.py by file path
script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'compute_additional_metrics.py'))
spec = importlib.util.spec_from_file_location('compute_additional_metrics', script_path)
cam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cam)
compute_metrics = cam.compute_metrics


def run_pair(real_path, synth_path, out_prefix):
    real = np.load(real_path)
    synth = np.load(synth_path)
    # Normalize shapes: collapse trailing dims to produce (N, L)
    if getattr(real, 'ndim', None) == 3:
        real = real.reshape(real.shape[0], -1)
    if getattr(synth, 'ndim', None) == 3:
        synth = synth.reshape(synth.shape[0], -1)
    metrics = compute_metrics(real, synth)
    out_dir = 'compare_out_test'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{out_prefix}_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(out_prefix, json.dumps(metrics))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pdf-real', required=True)
    p.add_argument('--pdf-synth', required=True)
    p.add_argument('--my-synth-af', required=False)
    p.add_argument('--my-synth-nsr', required=False)
    args = p.parse_args()

    # Run pdf real vs pdf synth
    run_pair(args.pdf_real, args.pdf_synth, 'pdf_real_vs_pdf_synth')

    # If provided, compare pdf real vs user's synths
    if args.my_synth_af:
        run_pair(args.pdf_real, args.my_synth_af, 'pdf_real_vs_my_synth_AF')
    if args.my_synth_nsr:
        run_pair(args.pdf_real, args.my_synth_nsr, 'pdf_real_vs_my_synth_NSR')


if __name__ == '__main__':
    main()
