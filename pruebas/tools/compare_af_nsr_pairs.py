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

# Paths
orig_af = Path(r'C:\Users\BISITE-NEL\Desktop\pruebas\1024seq_AF.orig.npy')
proc_af = Path(r'C:\Users\BISITE-NEL\Desktop\pruebas\1024seq_AF.npy')
orig_nsr = Path(r'C:\Users\BISITE-NEL\Desktop\pruebas\1024seq_NSR.orig.npy')
proc_nsr = Path(r'C:\Users\BISITE-NEL\Desktop\pruebas\1024seq_NSR.npy')
gen_af = Path('compare_out_generated/sssd_pulso_run1/sssd_AF_1024.npy')
gen_nsr = Path('compare_out_generated/sssd_pulso_run1/sssd_NSR_1024.npy')

for p in [orig_af, proc_af, orig_nsr, proc_nsr, gen_af, gen_nsr]:
    if not p.exists():
        print('Missing file:', p)
        sys.exit(2)

# Load and prepare
OrigAF = prepare(np.load(orig_af))
ProcAF = prepare(np.load(proc_af))
GenAF = prepare(np.load(gen_af))
OrigNSR = prepare(np.load(orig_nsr))
ProcNSR = prepare(np.load(proc_nsr))
GenNSR = prepare(np.load(gen_nsr))

# Align target lengths per pair (use min of the two)
def align_and_compute(A,B,label):
    t = min(A.shape[1], B.shape[1])
    Ar = resample_to(A, t)
    Br = resample_to(B, t)
    res = compute_all_metrics(Ar, Br, name=label)
    res.update({'label': label})
    return res

# AF table
rows_af = []
rows_af.append(align_and_compute(OrigAF, ProcAF, 'OrigAF_vs_ProcAF'))
rows_af.append(align_and_compute(OrigAF, GenAF, 'OrigAF_vs_GenAF'))

df_af = pd.DataFrame(rows_af)
outdir = Path('compare_out_generated')
outdir.mkdir(parents=True, exist_ok=True)
csv_af = outdir / 'AF_pair_metrics.csv'
html_af = outdir / 'AF_pair_metrics.html'
df_af.to_csv(csv_af, index=False)
try:
    styled = df_af.style.format(na_rep='-', formatter="{:.4f}").render()
    with open(html_af, 'w', encoding='utf-8') as fh:
        fh.write('<meta charset="utf-8">\n')
        fh.write('<h2>AF Pair Metrics</h2>\n')
        fh.write(styled)
except Exception:
    with open(html_af, 'w', encoding='utf-8') as fh:
        fh.write(df_af.to_html(index=False))

# NSR table
rows_nsr = []
rows_nsr.append(align_and_compute(OrigNSR, ProcNSR, 'OrigNSR_vs_ProcNSR'))
rows_nsr.append(align_and_compute(OrigNSR, GenNSR, 'OrigNSR_vs_GenNSR'))

df_nsr = pd.DataFrame(rows_nsr)
csv_nsr = outdir / 'NSR_pair_metrics.csv'
html_nsr = outdir / 'NSR_pair_metrics.html'
df_nsr.to_csv(csv_nsr, index=False)
try:
    styled = df_nsr.style.format(na_rep='-', formatter="{:.4f}").render()
    with open(html_nsr, 'w', encoding='utf-8') as fh:
        fh.write('<meta charset="utf-8">\n')
        fh.write('<h2>NSR Pair Metrics</h2>\n')
        fh.write(styled)
except Exception:
    with open(html_nsr, 'w', encoding='utf-8') as fh:
        fh.write(df_nsr.to_html(index=False))

print('Saved AF CSV/HTML:', csv_af, html_af)
print('Saved NSR CSV/HTML:', csv_nsr, html_nsr)
