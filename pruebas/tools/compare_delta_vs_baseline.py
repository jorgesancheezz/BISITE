import pandas as pd
from pathlib import Path
import numpy as np

outdir = Path('compare_out_generated')
csv_gen = outdir / 'orig_vs_generated_metrics.csv'
csv_base = outdir / 'orig_vs_orig_metrics.csv'
if not csv_gen.exists() or not csv_base.exists():
    print('Missing required CSVs:', csv_gen.exists(), csv_base.exists())
    raise SystemExit(1)

df_gen = pd.read_csv(csv_gen)
df_base = pd.read_csv(csv_base)

# baseline is single-row comparing AF vs NSR
base_row = df_base.iloc[0]

rows = []
for _, r in df_gen.iterrows():
    name = r.get('class', '') or r.get('class_pair','')
    out = {'name': name}
    for col in r.index:
        if col in ['class','pair']:
            out[col] = r[col]
            continue
        # skip non-numeric
        try:
            val = float(r[col]) if pd.notna(r[col]) else np.nan
        except Exception:
            out[col] = r[col]
            continue
        # baseline value: use same metric from baseline row if present
        base_val = base_row.get(col, np.nan) if col in base_row.index else np.nan
        try:
            base_val = float(base_val) if pd.notna(base_val) else np.nan
        except Exception:
            base_val = np.nan
        delta = val - base_val if (not np.isnan(val) and not np.isnan(base_val)) else np.nan
        pct = (delta / base_val * 100.0) if (not np.isnan(delta) and base_val != 0 and not np.isnan(base_val)) else np.nan
        out[col] = val
        out[col + '_baseline'] = base_val
        out[col + '_delta'] = delta
        out[col + '_pct'] = pct
    rows.append(out)

df_out = pd.DataFrame(rows)
# save
csvp = outdir / 'delta_vs_baseline.csv'
htmlp = outdir / 'delta_vs_baseline.html'
df_out.to_csv(csvp, index=False)
try:
    styled = df_out.style.format(na_rep='-', formatter="{:.4f}").render()
    with open(htmlp, 'w', encoding='utf-8') as fh:
        fh.write('<meta charset="utf-8">\n')
        fh.write('<h2>Delta vs Baseline (orig_vs_orig)</h2>\n')
        fh.write(styled)
except Exception:
    with open(htmlp, 'w', encoding='utf-8') as fh:
        fh.write(df_out.to_html(index=False))

print('Saved CSV:', csvp)
print('Saved HTML:', htmlp)
