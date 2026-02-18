import pandas as pd
from pathlib import Path

OUT_DIR = Path('compare_out_generated')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_if(path):
    p = Path(path)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None

def main():
    demo_metrics = read_if('compare_out_demo/metrics_master_table_demo_vs_pulso.csv')
    pulso_gen = read_if('compare_out_generated/pulso_vs_generated_metrics.csv')

    parts = []
    if demo_metrics is not None:
        demo_metrics = demo_metrics.copy()
        demo_metrics['source'] = 'demo_vs_pulso'
        parts.append(demo_metrics)
    if pulso_gen is not None:
        pg = pulso_gen.copy()
        pg['source'] = 'pulso_vs_generated'
        parts.append(pg)

    if not parts:
        print('No metric files found to aggregate.')
        return

    df = pd.concat(parts, ignore_index=True, sort=False)
    out_csv = OUT_DIR / 'aggregate_metrics.csv'
    out_html = OUT_DIR / 'aggregate_metrics.html'
    df.to_csv(out_csv, index=False)
    try:
        styled = df.style.format(na_rep='-', formatter="{:.4f}").render()
        with open(out_html, 'w', encoding='utf-8') as fh:
            fh.write('<meta charset="utf-8">\n')
            fh.write('<h2>Aggregate Metrics</h2>\n')
            fh.write(styled)
    except Exception:
        with open(out_html, 'w', encoding='utf-8') as fh:
            fh.write(df.to_html(index=False))

    print('Saved:', out_csv)
    print('Saved:', out_html)

if __name__ == '__main__':
    main()
