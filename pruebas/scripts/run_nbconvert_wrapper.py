"""Run nbconvert ExecutePreprocessor after setting WindowsSelectorEventLoopPolicy.

This avoids the zmq/Proactor event loop RuntimeWarning on Windows when using
nbconvert/zmq from an asyncio Proactor loop.

Usage:
    python run_nbconvert_wrapper.py --input compare_synthetic_vs_real_report.ipynb --output compare_executed.ipynb --timeout 1200
"""
import argparse
import asyncio
import warnings
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def set_selector_policy_if_windows():
    if sys.platform.startswith('win'):
        try:
            # suppress DeprecationWarnings about WindowsSelectorEventLoopPolicy
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            policy = asyncio.WindowsSelectorEventLoopPolicy()
            asyncio.set_event_loop_policy(policy)
        except Exception:
            # older Python or not available — ignore
            pass

def run_notebook(input_path, output_path, timeout):
    set_selector_policy_if_windows()
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input notebook path')
    parser.add_argument('--output', '-o', required=True, help='Output notebook path')
    parser.add_argument('--timeout', '-t', type=int, default=1200, help='ExecutePreprocessor timeout')
    args = parser.parse_args()
    run_notebook(args.input, args.output, args.timeout)

if __name__ == '__main__':
    main()
