import os
import traceback
try:
    import wfdb
except Exception as e:
    print('ERROR: wfdb no está instalado o no se puede importar:', e)
    raise SystemExit(1)

rec_rel = os.path.join('p10', 'p10001', 'p10001_s00')
rec_path = os.path.splitext(rec_rel)[0]
print('Reading record (base):', rec_path)
try:
    ann = wfdb.rdann(rec_path, 'atr')
    samples = getattr(ann, 'sample', None)
    symbols = getattr(ann, 'symbol', None)
    aux = getattr(ann, 'aux_note', None)
    print('n_annotations =', len(samples) if samples is not None else 'None')
    print('\nFirst 50 samples:')
    print(list(samples[:50]))
    print('\nFirst 50 symbols:')
    print(list(symbols[:50]) if symbols is not None else 'None')
    if aux is not None:
        print('\nFirst 50 aux_note entries (showing non-empty):')
        cnt = 0
        for i,a in enumerate(aux[:500]):
            if a and a.strip():
                print(i, repr(a))
                cnt += 1
            if cnt >= 50:
                break
    else:
        print('\naux_note not present')
except Exception:
    traceback.print_exc()
