import numpy as np
import math


def ensure_shape_npy(x, target=(1024, 3000, 1), lead_idx=0):
    import torch
    if isinstance(x, np.ndarray):
        arr = x
    elif 'torch' in globals() and torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        try:
            import torch as _t
            if _t.is_tensor(x):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)
        except Exception:
            arr = np.asarray(x)

    arr = np.asarray(arr)
    if arr.ndim == 3:
        N, C, T = arr.shape
        y = arr[:, lead_idx, :]
    elif arr.ndim == 2:
        N, T = arr.shape
        y = arr
    elif arr.ndim == 1:
        N = 1
        T = arr.shape[0]
        y = arr.reshape(1, T)
    else:
        raise ValueError("Unsupported array shape for ensure_shape_npy: {}".format(arr.shape))

    target_N, target_T, target_C = target

    if T != target_T:
        if target_T % T == 0:
            reps = target_T // T
            y = np.tile(y, (1, reps))
        else:
            xp = np.arange(T)
            x_new = np.linspace(0, T - 1, target_T)
            y_res = np.zeros((y.shape[0], target_T), dtype=y.dtype)
            for i in range(y.shape[0]):
                y_res[i] = np.interp(x_new, xp, y[i])
            y = y_res

    if y.shape[0] < target_N:
        times = int(math.ceil(target_N / y.shape[0]))
        y = np.tile(y, (times, 1))[:target_N]
    else:
        y = y[:target_N]

    y = y.reshape((target_N, target_T, target_C))
    return y.astype(np.float32)


if __name__ == '__main__':
    dummy = np.random.randn(4, 8, 1000).astype(np.float32)
    arr = ensure_shape_npy(dummy, target=(1024,3000,1), lead_idx=0)
    out = 'scripts/test_1024_3000_1_standalone.npy'
    np.save(out, arr)
    print('saved', out, 'shape=', arr.shape)
