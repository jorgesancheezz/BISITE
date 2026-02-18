# PULSOVITAL – Ejecutar todo con `-m`

Este paquete está organizado en submódulos (`core/`, `data/`, `metrics/`, `plotting/`, `training/`).
Todas las herramientas se ejecutan mediante `python -m PULSOVITAL.<subpaquete>.<módulo>`.

Requisitos mínimos: Python 3.10+, NumPy, SciPy, Matplotlib, PyTorch. Opcional: scikit-learn (para `training/train.py`), PyYAML (para exportar args), Plotly (HTML interactivo).

## Comandos rápidos

- Ayudas (todas muestran `--help`):
  - `python -m PULSOVITAL.data.synthetic_loader --help`
  - `python -m PULSOVITAL.metrics.compare_stats --help`
  - `python -m PULSOVITAL.metrics.fid_with_model --help`
  - `python -m PULSOVITAL.metrics.fid_compare_both --help`
  - `python -m PULSOVITAL.plotting.plot_psd_full --help`
  - `python -m PULSOVITAL.plotting.plot_fid_csv --help`
  - `python -m PULSOVITAL.plotting.plot_compare_mean_std --help`
  - `python -m PULSOVITAL.training.sim_train --help`
  - `python -m PULSOVITAL.training.similarity_train --help`
  - `python -m PULSOVITAL.training.train --help` (importa dependencias opcionales)

## Flujo mínimo validado (ejemplo)

1) Generar datos sintéticos

- Referencia (alpha=0.3):
  `python -m PULSOVITAL.data.synthetic_loader --generator wavelet --length 1000 --alpha 0.3 --noise 0.05 --num-samples 256 --out PULSOVITAL/results/synth_ref.npy`

- Comparación (alpha=0.8):
  `python -m PULSOVITAL.data.synthetic_loader --generator wavelet --length 1000 --alpha 0.8 --noise 0.05 --num-samples 256 --out PULSOVITAL/results/synth_cmp.npy`

2) Comparar estadísticas + FID (características de espectrograma)

`python -m PULSOVITAL.metrics.compare_stats --ref PULSOVITAL/results/synth_ref.npy --cmp PULSOVITAL/results/synth_cmp.npy --out-json PULSOVITAL/results/compare_synth_ref_vs_cmp.json`

3) Graficar PSD (promedio y comparativas)

`python -m PULSOVITAL.plotting.plot_psd_full --files PULSOVITAL/results/synth_ref.npy PULSOVITAL/results/synth_cmp.npy --out-dir PULSOVITAL/results/comparison_plots_synth`

4) FID con embeddings de modelo (si existe `PULSOVITAL/results/similarity_model.pt`)

`python -m PULSOVITAL.metrics.fid_with_model pair --ckpt PULSOVITAL/results/similarity_model.pt --ref-generator wavelet --ref-alpha 0.3 --cmp-generator wavelet --cmp-alpha 0.8 --length 1000 --samples 256 --num-workers 0 --batch-size 64`

5) Barrido combinado FID (espectrograma vs modelo) a CSV + gráficas

- CSV:
`python -m PULSOVITAL.metrics.fid_compare_both --ckpt PULSOVITAL/results/similarity_model.pt --ref-generator wavelet --ref-alpha 0.3 --cmp-generator wavelet --alpha-start 0.3 --alpha-stop 0.9 --alpha-step 0.3 --length 1000 --samples 256 --batch-size 64 --num-workers 0 --out-csv PULSOVITAL/results/fid_compare_both_test.csv`

- PNG (x=series FID, líneas=alpha):
`python -m PULSOVITAL.plotting.plot_fid_csv --csv PULSOVITAL/results/fid_compare_both_test.csv --x-axis fid`

- Media±desv:
`python -m PULSOVITAL.plotting.plot_compare_mean_std --csv PULSOVITAL/results/fid_compare_both_test.csv`

## Notas sobre entrenamiento

- `PULSOVITAL.training.sim_train` y `PULSOVITAL.training.similarity_train` son auto-contenidos y no requieren dependencias externas.
- `PULSOVITAL.training.train` depende de un paquete externo `diffusion` (DiT). Se ha hecho opcional su importación para que `--help` funcione. Si vas a entrenar con ese script:
  - Instala el paquete `diffusion` correspondiente, `pyyaml` y `scikit-learn`.
  - Requiere GPU (usa DDP/NCCL).
  - Alternativa: usa `similarity_train` para un entrenamiento ligero de un encoder 1D.

## Limpieza de raíz

Los shims de raíz no son necesarios; ya puedes ejecutar todo con `-m`. Si deseas limpiar archivos sueltos en la raíz, asegúrate antes de migrar tus alias/comandos a la forma `python -m PULSOVITAL.<...>`.
