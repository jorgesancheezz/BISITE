import glob, os, numpy as np

def inspect(folder):
    pattern = os.path.join(folder, '*', 'signal.npy')
    files = sorted(glob.glob(pattern))
    print(f"Folder: {folder} - found {len(files)} files")
    for i,f in enumerate(files[:10]):
        try:
            a = np.load(f)
            print(i, f, a.shape, a.dtype)
        except Exception as e:
            print('ERR', f, e)
    if len(files)>10:
        print('...')
    return files

if __name__=='__main__':
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    af = os.path.join(base, 'PULSOVITAL', 'npy_output', 'AF')
    nsr = os.path.join(base, 'PULSOVITAL', 'npy_output', 'NSR')
    files_af = inspect(af)
    files_nsr = inspect(nsr)
    # print one sample stats if present
    if files_af:
        a = np.load(files_af[0])
        print('AF sample min/max/mean:', float(a.min()), float(a.max()), float(a.mean()))
    if files_nsr:
        b = np.load(files_nsr[0])
        print('NSR sample min/max/mean:', float(b.min()), float(b.max()), float(b.mean()))
    
    # write lists to temp for further steps
    import json, tempfile
    out = {'af_files': files_af, 'nsr_files': files_nsr}
    fname = os.path.join(tempfile.gettempdir(), 'npy_signal_lists.json')
    with open(fname, 'w', encoding='utf-8') as fh:
        json.dump(out, fh)
    print('Wrote file list to', fname)
