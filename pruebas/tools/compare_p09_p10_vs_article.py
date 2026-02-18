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
p09_af = Path('PULSOVITAL/npy_output_p09_consolidated/1024seq_AF.npy')
p09_nsr = Path('PULSOVITAL/npy_output_p09_consolidated/1024seq_NSR.npy')
p10_af = Path('PULSOVITAL/Metricas/1024seq_AF.npy')
p10_nsr = Path('PULSOVITAL/Metricas/1024seq_NSR.npy')
article_af = Path('compare_out_generated/sssd_pulso_run1/sssd_AF_1024.npy')
article_nsr = Path('compare_out_generated/sssd_pulso_run1/sssd_NSR_1024.npy')

for p in [p09_af,p09_nsr,p10_af,p10_nsr,article_af,article_nsr]:
    if not p.exists():
        print('Missing file:', p)
        sys.exit(2)

# load
R_p09_af = prepare(np.load(p09_af))
R_p09_nsr = prepare(np.load(p09_nsr))
R_p10_af = prepare(np.load(p10_af))
R_p10_nsr = prepare(np.load(p10_nsr))
G_af = prepare(np.load(article_af))
G_nsr = prepare(np.load(article_nsr))

rows = []
# P09 AF vs Article AF
rows.append({'pair':'P09AF_vs_ArticleAF', **compute_all_metrics(resample_to(R_p09_af, min(R_p09_af.shape[1], G_af.shape[1])), resample_to(G_af, min(R_p09_af.shape[1], G_af.shape[1])), 'P09AF_vs_ArticleAF')})
# P10 AF vs Article AF
rows.append({'pair':'P10AF_vs_ArticleAF', **compute_all_metrics(resample_to(R_p10_af, min(R_p10_af.shape[1], G_af.shape[1])), resample_to(G_af, min(R_p10_af.shape[1], G_af.shape[1])), 'P10AF_vs_ArticleAF')})
# P09 NSR vs Article NSR
rows.append({'pair':'P09NSR_vs_ArticleNSR', **compute_all_metrics(resample_to(R_p09_nsr, min(R_p09_nsr.shape[1], G_nsr.shape[1])), resample_to(G_nsr, min(R_p09_nsr.shape[1], G_nsr.shape[1])), 'P09NSR_vs_ArticleNSR')})
# P10 NSR vs Article NSR
rows.append({'pair':'P10NSR_vs_ArticleNSR', **compute_all_metrics(resample_to(R_p10_nsr, min(R_p10_nsr.shape[1], G_nsr.shape[1])), resample_to(G_nsr, min(R_p10_nsr.shape[1], G_nsr.shape[1])), 'P10NSR_vs_ArticleNSR')})

df = pd.DataFrame(rows)
outdir = Path('compare_out_generated')
outdir.mkdir(parents=True, exist_ok=True)
csvp = outdir / 'p09_p10_vs_article_metrics.csv'
htmlp = outdir / 'p09_p10_vs_article_metrics.html'
df.to_csv(csvp, index=False)
try:
    styled = df.style.format(na_rep='-', formatter="{:.4f}").render()
    with open(htmlp, 'w', encoding='utf-8') as fh:
        fh.write('<meta charset="utf-8">\n')
        fh.write('<h2>P09/P10 vs Article Metrics</h2>\n')
        fh.write(styled)
except Exception:
    with open(htmlp, 'w', encoding='utf-8') as fh:
        fh.write(df.to_html(index=False))

print('Saved CSV:', csvp)
print('Saved HTML:', htmlp)
