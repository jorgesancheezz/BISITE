import sys
import numpy as np
from compute_additional_metrics import compute_metrics

np.random.seed(0)

def assert_close(a, b, tol=1e-6, name=None):
    if abs(a-b) <= tol:
        print(f"OK {name}: {a} ~ {b}")
        return True
    print(f"FAIL {name}: {a} != {b} (tol {tol})")
    return False


def test_identical():
    print('Test: identical datasets')
    real = np.random.randn(50, 200)
    synth = real.copy()
    metrics = compute_metrics(real, synth)
    # MDD, ACD, SD, KD should be ~0
    ok = True
    ok &= assert_close(metrics['MDD_wasserstein'], 0.0, tol=1e-12, name='MDD')
    ok &= assert_close(metrics['ACD_mean_abs'], 0.0, tol=1e-12, name='ACD')
    ok &= assert_close(metrics['SD'], 0.0, tol=1e-12, name='SD')
    ok &= assert_close(metrics['KD'], 0.0, tol=1e-12, name='KD')
    # DS should be ~0.5
    ds = metrics['DS']
    if 0.45 <= ds <= 0.55:
        print(f"OK DS: {ds}")
    else:
        print(f"WARN DS outside [0.45,0.55]: {ds}")
        ok = False
    # PS should be numeric between 0 and 1
    ps = metrics['PS']
    if ps is None or (isinstance(ps, float) and (np.isnan(ps) or ps<0 or ps>1)):
        print(f"WARN PS unexpected: {ps}")
        # not fail
    else:
        print(f"OK PS: {ps}")
    return ok


def test_shifted():
    print('\nTest: shifted datasets (synth = real + 3)')
    real = np.random.randn(50, 200)
    synth = real + 3.0
    metrics = compute_metrics(real, synth)
    ok = True
    if metrics['MDD_wasserstein'] > 0.1:
        print(f"OK MDD > 0: {metrics['MDD_wasserstein']}")
    else:
        print(f"FAIL MDD not >0.1: {metrics['MDD_wasserstein']}")
        ok = False
    if metrics['SD'] >= 0.0:
        print(f"OK SD: {metrics['SD']}")
    if metrics['KD'] >= 0.0:
        print(f"OK KD: {metrics['KD']}")
    if metrics['ACD_mean_abs'] >= 0.0:
        print(f"OK ACD: {metrics['ACD_mean_abs']}")
    return ok


def main():
    all_ok = True
    all_ok &= test_identical()
    all_ok &= test_shifted()
    if all_ok:
        print('\nAll validation tests passed')
        sys.exit(0)
    else:
        print('\nSome validation tests failed')
        sys.exit(2)

if __name__ == '__main__':
    main()
