import sys
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
compare_mod_path = repo_root / 'tools' / 'compare_with_article_demos.py'
if not compare_mod_path.exists():
    print('Missing compare_with_article_demos.py at', compare_mod_path)
    sys.exit(1)

spec = importlib.util.spec_from_file_location('compare_demo', str(compare_mod_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

prepare = getattr(mod, 'prepare')
resample_to = getattr(mod, 'resample_to')
compute_all_metrics = getattr(mod, 'compute_all_metrics')

orig_af = Path(r'C:\Users\BISITE-NEL\Desktop\pruebas\1024seq_AF.orig.npy')
orig_nsr = Path(r'C:\Users\BISITE-NEL\Desktop\pruebas\1024seq_NSR.orig.npy')
for p in [orig_af, orig_nsr]:
    if not p.exists():
        print('Missing file:', p)
        sys.exit(2)

A_af = prepare(np.load(orig_af))
A_nsr = prepare(np.load(orig_nsr))

# same-file comparisons
res_af = compute_all_metrics(A_af, A_af, name='Self_Af')
res_af.update({'pair':'AF_self'})
res_nsr = compute_all_metrics(A_nsr, A_nsr, name='Self_Nsr')
res_nsr.update({'pair':'NSR_self'})

outdir = Path('compare_out_generated')
outdir.mkdir(parents=True, exist_ok=True)

pd.DataFrame([res_af]).to_csv(outdir / 'AF_self_metrics.csv', index=False)
try:
    styled = pd.DataFrame([res_af]).style.format(na_rep='-', formatter="{:.4f}").render()
    with open(outdir / 'AF_self_metrics.html', 'w', encoding='utf-8') as fh:
        fh.write('<meta charset="utf-8">\n')
        fh.write('<h2>AF Self Comparison</h2>\n')
        fh.write(styled)
except Exception:
    pd.DataFrame([res_af]).to_html(outdir / 'AF_self_metrics.html')

pd.DataFrame([res_nsr]).to_csv(outdir / 'NSR_self_metrics.csv', index=False)
try:
    styled = pd.DataFrame([res_nsr]).style.format(na_rep='-', formatter="{:.4f}").render()
    with open(outdir / 'NSR_self_metrics.html', 'w', encoding='utf-8') as fh:
        fh.write('<meta charset="utf-8">\n')
        fh.write('<h2>NSR Self Comparison</h2>\n')
        fh.write(styled)
except Exception:
    pd.DataFrame([res_nsr]).to_html(outdir / 'NSR_self_metrics.html')

print('Saved AF_self_metrics.csv/html and NSR_self_metrics.csv/html in', outdir)
