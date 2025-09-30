import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from tqdm import tqdm
import warnings
import wfdb
import glob

# Ocultar warnings de NeuroKit2 y otros
warnings.filterwarnings("ignore")

# Definir rutas a los datos preparados y carpeta de salida de gráficos
data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resultados2'))
output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resultados_plot'))
os.makedirs(output_folder, exist_ok=True)

# Leer datos ya preparados (señales y estadísticas)
X = np.load(os.path.join(data_folder, 'X_ml.npy'))  # (n_registros, max_len)
df = pd.read_csv(os.path.join(data_folder, 'estadisticas_signales.csv'))

# Filtrar solo registros con regiones NSR detectadas
if 'is_nsr' in df.columns:
    mask_nsr = df['is_nsr'] == True
    df_nsr = df[mask_nsr].reset_index(drop=True)
    X_nsr = X[mask_nsr.values]
    print(f"Se encontraron {len(df_nsr)} registros con regiones NSR.")
else:
    print("No se encontró la columna 'is_nsr'. Usando todos los registros.")
    df_nsr = df
    X_nsr = X

# Función para extraer los puntos de la señal dentro de regiones NSR
def extraer_segmentos_nsr(record_path, signal_len):
    try:
        ann = wfdb.rdann(record_path, 'atr')
        aux = getattr(ann, 'aux_note', None)
        samples = getattr(ann, 'sample', None)
        if aux is None or samples is None:
            return []
        segmentos = []
        inside_nsr = False
        start = None
        for i, note in enumerate(aux):
            if note and note.startswith('(N'):
                inside_nsr = True
                start = samples[i]
            elif note and note == ')' and inside_nsr:
                end = samples[i]
                # Limitar a la longitud de la señal
                if start is not None and end > start:
                    segmentos.append((max(0, start), min(signal_len, end)))
                inside_nsr = False
                start = None
        return segmentos
    except Exception:
        return []

# Parámetros de optimización para la visualización
N_MUESTRAS = 30  # Solo analizamos/graficamos las primeras N muestras para acelerar
MIN_STD = 0.05     # Solo analizamos señales con suficiente variabilidad

# Graficar y analizar solo los puntos dentro de regiones NSR
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'p10'))
for i in tqdm(range(X_nsr.shape[0]), desc="Graficando solo regiones NSR"):
    record = df_nsr.iloc[i]['record'] if 'record' in df_nsr.columns else f"registro_{i}"
    fs = df_nsr.iloc[i]['fs'] if 'fs' in df_nsr.columns else 500  # valor por defecto
    # Ruta al registro original para anotaciones
    rec_path = os.path.join(base_path, record)
    rec_path = os.path.splitext(rec_path)[0]
    # Extraer segmentos NSR
    segmentos = extraer_segmentos_nsr(rec_path, X_nsr.shape[1])
    if not segmentos:
        continue
    signal = X_nsr[i]  # Definir la señal original para este registro
    arrays = [signal[start:end] for start, end in segmentos if end > start]
    if not arrays or sum(len(a) for a in arrays) < 2:
        continue
    nsr_points = np.concatenate(arrays)
    if len(nsr_points) < 2 or np.std(nsr_points) < MIN_STD:
        continue
    # Detectar picos R sobre la señal NSR
    try:
        ecg_peaks, info = nk.ecg_peaks(nsr_points, sampling_rate=fs)
        r_peaks = info["ECG_R_Peaks"]
        heart_rate = nk.ecg_rate(ecg_peaks, sampling_rate=fs)
        hr_mean = heart_rate.mean() if len(heart_rate) > 0 else np.nan
        hr_std = heart_rate.std() if len(heart_rate) > 0 else np.nan
    except Exception:
        hr_mean = hr_std = np.nan
        r_peaks = []
    # Graficar solo los puntos NSR
    plt.figure(figsize=(12,4))
    plt.plot(nsr_points, label="ECG NSR")
    r_peaks_in_window = r_peaks[r_peaks < len(nsr_points)] if len(r_peaks) > 0 else []
    plt.scatter(r_peaks_in_window, nsr_points[r_peaks_in_window], color='red', marker='x', label='R Peaks')
    plt.title(f"Registro {record} - solo regiones NSR")
    plt.xlabel("Muestra (solo NSR)")
    plt.ylabel("Amplitud")
    plt.legend()
    img_path = os.path.join(output_folder, f"{record}_soloNSR.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)
    plt.close()
    # Guardar métricas en el DataFrame
    df_nsr.loc[i, 'hr_mean_nsr'] = hr_mean
    df_nsr.loc[i, 'hr_std_nsr'] = hr_std
# Guardar CSV actualizado solo para registros NSR
df_nsr.to_csv(os.path.join(output_folder, "metrics_con_hr_nsr_soloNSR.csv"), index=False)
print("Visualización completada SOLO para puntos dentro de regiones NSR. CSV y gráficos guardados en 'resultados_plot'.")
