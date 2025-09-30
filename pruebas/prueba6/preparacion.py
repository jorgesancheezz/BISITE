import neurokit2 as nk
import wfdb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import scipy.io as sio

# Carpetas y parámetros (volver a p10 como raíz fija de datos)
SCRIPT_DIR = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'p10'))
output_folder = os.path.abspath(os.path.join(SCRIPT_DIR, 'resultados2'))
max_len = 5000  # longitud objetivo de cada segmento (relleno o recorte)
os.makedirs(output_folder, exist_ok=True)
list_of_signals_fixed = []
eda_df = pd.DataFrame()

# Buscar todos los .hea dentro de p10 de forma recursiva
glob_pattern = os.path.join(base_path, '**', '*.hea')
hea_files = glob.glob(glob_pattern, recursive=True)
if not hea_files:
    print(f"No se encontraron .hea bajo: {base_path}")
    signals_array = np.array(list_of_signals_fixed)
    print("Array de señales con forma:", signals_array.shape)
    eda_df.to_csv(os.path.join(output_folder, "estadisticas_signales.csv"), index=False)
    print("Estadísticas guardadas en 'estadisticas_signales.csv'")
    print(eda_df)
    print("No se encontraron registros. Nada que procesar.")
    raise SystemExit(0)

# Obtener nombres base de registro (sin extensión)
record_bases = [os.path.splitext(f)[0] for f in hea_files]

skipped = 0
for rec_path in tqdm(record_bases, desc="Procesando registros"):
    try:
        r = wfdb.rdrecord(rec_path)
        # Validar que haya señal analógica
        if getattr(r, 'p_signal', None) is None:
            raise ValueError("registro sin p_signal")
        signal = r.p_signal[:, 0]  # primer canal
        n_sig = r.n_sig
        sig_len = r.sig_len
        fs_used = r.fs
    except Exception as e:
        # Fallback para datasets tipo PhysioNet 2017 (MAT + HEA con 'val')
        mat_path = rec_path + '.mat'
        hea_path = rec_path + '.hea'
        try:
            if not os.path.exists(mat_path) or not os.path.exists(hea_path):
                raise FileNotFoundError("faltan .mat o .hea")
            # Leer fs de la primera línea del .hea (formato: name n_sig fs n_samples ...)
            with open(hea_path, 'r') as hf:
                header_first = hf.readline().strip()
            toks = header_first.split()
            fs_used = float(toks[2]) if len(toks) >= 3 else 300.0
            # Leer .mat con variable 'val'
            mat = sio.loadmat(mat_path)
            val = mat.get('val')
            if val is None:
                raise ValueError(".mat sin variable 'val'")
            # val típico: (n_leads, n_samples). Usamos primer canal
            if val.ndim == 2:
                n_sig = val.shape[0]
                sig_len = val.shape[1]
                signal = val[0, :].astype(float)
            else:
                arr = np.array(val).squeeze()
                n_sig = 1
                sig_len = arr.size
                signal = arr.astype(float)
        except Exception as e2:
            skipped += 1
            print(f"Saltando {os.path.relpath(rec_path, base_path)}: {e} | fallback: {e2}")
            continue

    # Recortar la señal antes de limpiar
    if len(signal) > max_len:
        signal = signal[:max_len]
    # Limpiar ECG
    clean_signal = nk.ecg_clean(signal, sampling_rate=fs_used)
    # Normalización segura (evitar división por cero)
    sig_std = np.std(clean_signal)
    signal_norm = (clean_signal - np.mean(clean_signal)) / sig_std if sig_std > 0 else (clean_signal - np.mean(clean_signal))

    if len(signal_norm) > max_len:
        signal_fixed = signal_norm[:max_len]
    else:
        signal_fixed = np.pad(signal_norm, (0, max_len - len(signal_norm)), 'constant')

    list_of_signals_fixed.append(signal_fixed)

    # Guardar estadísticas en DataFrame
    new_row = pd.DataFrame([{
        "record": os.path.relpath(rec_path, base_path),
        "n_signals": n_sig,
        "length_samples": sig_len,
        "fs": fs_used,
        "duration_sec": sig_len / fs_used,
        "mean_amplitude": np.mean(signal),
        "std_amplitude": np.std(signal),
    }])
    eda_df = pd.concat([eda_df, new_row], ignore_index=True)

signals_array = np.array(list_of_signals_fixed)
print("Array de señales con forma:", signals_array.shape)
eda_df.to_csv(os.path.join(output_folder, "estadisticas_signales.csv"), index=False)
print("Estadísticas guardadas en 'estadisticas_signales.csv'")
print(eda_df.head())
print(f"Registros procesados: {len(list_of_signals_fixed)} | Saltados: {skipped}")

if signals_array.shape[0] == 0:
    print("No se encontraron registros. Nada que procesar.")
    raise SystemExit(0)

# Guardar tensor para ML
np.save(os.path.join(output_folder, "X_ml.npy"), signals_array)
print("Tensor de señales guardado en 'X_ml.npy'")

