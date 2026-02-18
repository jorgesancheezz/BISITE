import os
import numpy as np
from pathlib import Path
import pandas as pd
import importlib.util


def load_compare_module(path):
    spec = importlib.util.spec_from_file_location('compare_demo', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    repo_root = Path(__file__).resolve().parent.parent
    compare_mod_path = repo_root / 'tools' / 'compare_with_article_demos.py'
    if not compare_mod_path.exists():
        print('Cannot find compare_with_article_demos.py at', compare_mod_path)
        return
    mod = load_compare_module(str(compare_mod_path))

    # Expect these functions in the module
    compute_all_metrics = getattr(mod, 'compute_all_metrics')
    prepare = getattr(mod, 'prepare')
    resample_to = getattr(mod, 'resample_to')

    # Paths
    real_af = Path('PULSOVITAL/npy_output_p09_consolidated/1024seq_AF.npy')
    real_nsr = Path('PULSOVITAL/npy_output_p09_consolidated/1024seq_NSR.npy')
    gen_af = Path('compare_out_generated/sssd_pulso_run1/sssd_AF_1024.npy')
    gen_nsr = Path('compare_out_generated/sssd_pulso_run1/sssd_NSR_1024.npy')

    for p in [real_af, real_nsr, gen_af, gen_nsr]:
        if not p.exists():
            print('Missing file:', p)
            return

    R_af = prepare(np.load(real_af))
    R_nsr = prepare(np.load(real_nsr))
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
    af_res = compute_all_metrics(R_af_rs, G_af_rs, name='RealAF_vs_GenAF')
    af_res.update({'class': 'AF', 'pair': 'real_vs_gen'})
    rows.append(af_res)

    nsr_res = compute_all_metrics(R_nsr_rs, G_nsr_rs, name='RealNSR_vs_GenNSR')
    nsr_res.update({'class': 'NSR', 'pair': 'real_vs_gen'})
    rows.append(nsr_res)

    df = pd.DataFrame(rows)
    outdir = Path('compare_out_generated')
    outdir.mkdir(parents=True, exist_ok=True)
    csvp = outdir / 'pulso_vs_generated_metrics.csv'
    htmlp = outdir / 'pulso_vs_generated_metrics.html'
    df.to_csv(csvp, index=False)
    try:
        styled = df.style.format(na_rep='-', formatter="{:.4f}").render()
        with open(htmlp, 'w', encoding='utf-8') as fh:
            fh.write('<meta charset="utf-8">\n')
            fh.write('<h2>PULSOVITAL Real vs Generated Metrics</h2>\n')
            fh.write(styled)
    except Exception:
        with open(htmlp, 'w', encoding='utf-8') as fh:
            fh.write(df.to_html(index=False))

    print('Saved CSV:', csvp)
    print('Saved HTML:', htmlp)


if __name__ == '__main__':
    main()
