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

# Paths provided by the user
p09_af = Path('PULSOVITAL/npy_output_p09_consolidated/AF_processed_1024x3000x1.npy')
p09_nsr = Path('PULSOVITAL/npy_output_p09_consolidated/NSR_processed_1024x3000x1.npy')

# Candidate article outputs (fall back through these)
article_af_candidates = [
    Path('compare_out_generated/sssd_pulso_run1/sssd_AF_1024.npy'),
    Path('compare_out_generated/sssd_pulso_run2/sssd_AF_1024.npy'),
    Path('compare_out_generated/sssd_AF_1024.npy')
]
article_nsr_candidates = [
    Path('compare_out_generated/sssd_pulso_run1/sssd_NSR_1024.npy'),
    Path('compare_out_generated/sssd_pulso_run2/sssd_NSR_1024.npy'),
    Path('compare_out_generated/sssd_NSR_1024.npy')
]

def pick_existing(cands):
    for p in cands:
        if p.exists():
            return p
    return None

af_art = pick_existing(article_af_candidates)
nsr_art = pick_existing(article_nsr_candidates)

for p in [p09_af, p09_nsr]:
    if not p.exists():
        print('Missing P09 processed file:', p)
        sys.exit(2)

if af_art is None or nsr_art is None:
    print('Missing article-generated AF/NSR files. Checked candidates:')
    for p in article_af_candidates + article_nsr_candidates:
        print(' ', p)
    sys.exit(3)

print('Using article AF:', af_art)
print('Using article NSR:', nsr_art)

# load and prepare
R_af = prepare(np.load(p09_af))
R_nsr = prepare(np.load(p09_nsr))
G_af = prepare(np.load(af_art))
G_nsr = prepare(np.load(nsr_art))

rows = []
def pair_and_compute(R, G, label):
    L = min(R.shape[1], G.shape[1])
    Rr = resample_to(R, L)
    Gr = resample_to(G, L)
    return compute_all_metrics(Rr, Gr, label)

rows.append({'pair':'P09_processed_AF_vs_Article_AF', **pair_and_compute(R_af, G_af, 'P09AF_vs_ArticleAF')})
rows.append({'pair':'P09_processed_NSR_vs_Article_NSR', **pair_and_compute(R_nsr, G_nsr, 'P09NSR_vs_ArticleNSR')})

df = pd.DataFrame(rows)
outdir = Path('compare_out_generated')
outdir.mkdir(parents=True, exist_ok=True)
csvp = outdir / 'p09_processed_vs_article_metrics.csv'
htmlp = outdir / 'p09_processed_vs_article_metrics.html'
df.to_csv(csvp, index=False)
try:
    styled = df.style.format(na_rep='-', formatter="{:.4f}").render()
    with open(htmlp, 'w', encoding='utf-8') as fh:
        fh.write('<meta charset="utf-8">\n')
        fh.write('<h2>P09 Processed vs Article Metrics</h2>\n')
        fh.write(styled)
except Exception:
    with open(htmlp, 'w', encoding='utf-8') as fh:
        fh.write(df.to_html(index=False))

print('Saved CSV:', csvp)
print('Saved HTML:', htmlp)
