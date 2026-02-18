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

# Paths (user-provided originals)
orig_nsr = Path(r'C:\Users\BISITE-NEL\Desktop\pruebas\1024seq_NSR.orig.npy')
orig_af = Path(r'C:\Users\BISITE-NEL\Desktop\pruebas\1024seq_AF.orig.npy')
# Generated (defaults)
gen_af = Path('compare_out_generated/sssd_pulso_run1/sssd_AF_1024.npy')
gen_nsr = Path('compare_out_generated/sssd_pulso_run1/sssd_NSR_1024.npy')

for p in [orig_af, orig_nsr, gen_af, gen_nsr]:
    if not p.exists():
        print('Missing file:', p)
        sys.exit(2)

R_af = prepare(np.load(orig_af))
R_nsr = prepare(np.load(orig_nsr))
G_af = prepare(np.load(gen_af))
G_nsr = prepare(np.load(gen_nsr))

# align lengths
lens = [R_af.shape[1], R_nsr.shape[1], G_af.shape[1], G_nsr.shape[1]]
target_len = int(min(lens))
R_af_rs = resample_to(R_af, target_len)
R_nsr_rs = resample_to(R_nsr, target_len)
G_af_rs = resample_to(G_af, target_len)
G_nsr_rs = resample_to(G_nsr, target_len)

rows = []
af_res = compute_all_metrics(R_af_rs, G_af_rs, name='OrigAF_vs_GenAF')
af_res.update({'class': 'AF', 'pair': 'orig_vs_gen'})
rows.append(af_res)
nsr_res = compute_all_metrics(R_nsr_rs, G_nsr_rs, name='OrigNSR_vs_GenNSR')
nsr_res.update({'class': 'NSR', 'pair': 'orig_vs_gen'})
rows.append(nsr_res)

df = pd.DataFrame(rows)
outdir = Path('compare_out_generated')
outdir.mkdir(parents=True, exist_ok=True)
csvp = outdir / 'orig_vs_generated_metrics.csv'
htmlp = outdir / 'orig_vs_generated_metrics.html'
df.to_csv(csvp, index=False)
try:
    styled = df.style.format(na_rep='-', formatter="{:.4f}").render()
    with open(htmlp, 'w', encoding='utf-8') as fh:
        fh.write('<meta charset="utf-8">\n')
        fh.write('<h2>Originals vs Generated Metrics</h2>\n')
        fh.write(styled)
except Exception:
    with open(htmlp, 'w', encoding='utf-8') as fh:
        fh.write(df.to_html(index=False))

print('Saved CSV:', csvp)
print('Saved HTML:', htmlp)
