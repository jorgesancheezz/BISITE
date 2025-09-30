import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), 'resultados2')
CSV_PATH = os.path.join(DATA_DIR, 'estadisticas_signales.csv')
NPY_PATH = os.path.join(DATA_DIR, 'X_ml.npy')

df_signales = pd.read_csv(CSV_PATH)
X = np.load(NPY_PATH)

# Filtrar por FA si existe, si no por NSR, si no sin filtrar
if 'is_af' in df_signales.columns and df_signales['is_af'].any():
    df_signales = df_signales[df_signales['is_af'] == True].reset_index(drop=True)
    X = X[df_signales.index]
elif 'is_nsr' in df_signales.columns and df_signales['is_nsr'].any():
    df_signales = df_signales[df_signales['is_nsr'] == True].reset_index(drop=True)
    X = X[df_signales.index]

FS = int(df_signales['fs'].iloc[0])

def analizar_senal(senal):
    # Centrar la señal en cero
    senal = senal - np.mean(senal)
    # Normalizar la señal a varianza 1
    std = np.std(senal)
    if std > 0:
        senal = senal / std
    # Calcular autocorrelación de la señal centrada y normalizada
    autocorr = np.correlate(senal, senal, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Nos quedamos con la mitad positiva
    # Normalizar la autocorrelación a [-1, 1]
    max_abs = np.max(np.abs(autocorr))
    if max_abs > 0:
        autocorr = autocorr / max_abs
    # Buscar picos en la autocorrelación
    peaks, _ = find_peaks(autocorr, height=0.1, distance=int(0.3*FS))
    if len(peaks) < 2:
        # Retornar 5 valores siempre
        return np.nan, np.nan, autocorr, np.nan, np.nan
    # Calcular todos los RR intervalos entre picos consecutivos
    rr_lags = np.diff(peaks)
    rr_intervalos = rr_lags / FS  # en segundos
    if len(rr_intervalos) == 0:
        return np.nan, np.nan, autocorr, np.nan, np.nan
    # Usar el primer RR para la columna principal (como antes)
    rr_intervalo = rr_intervalos[0]
    hr_bpm = 60.0 / rr_intervalo if rr_intervalo > 0 else np.nan
    # Calcular media y std de HR usando todos los RR
    hrs = 60.0 / rr_intervalos
    mean_hr = np.mean(hrs) if len(hrs) > 0 else np.nan
    std_hr = np.std(hrs) if len(hrs) > 0 else np.nan
    return rr_intervalo, hr_bpm, autocorr, mean_hr, std_hr

# Elegir 300 índices aleatorios para graficar
num_graficos = 30
total = len(df_signales)
random.seed(42)
indices_graficar = set(random.sample(range(total), min(num_graficos, total)))
fecha_str = datetime.now().strftime('%H%M%S%d%m%Y')
PLOTS_DIR = os.path.join(DATA_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

resultados = []
start_total = time.time()

# Barra de progreso para mostrar avance y tiempo estimado
with tqdm(total=total, desc="Procesando señales", unit="senal") as pbar:
    for idx, row in df_signales.iterrows():
        # Identificador del segmento
        record = row['record'] if 'record' in row else f'senal_{idx}'
        try:
            senal = X[idx]  # Señal correspondiente
            rr_intervalo, hr_bpm, autocorr, mean_hr, std_hr = analizar_senal(senal)
            resultados.append([record, rr_intervalo, hr_bpm, mean_hr, std_hr])
            # Graficar solo si el índice está en la muestra aleatoria
            if idx in indices_graficar:
                plt.figure(figsize=(10,4))
                t = np.arange(len(senal))/FS  # Eje x para la señal
                lag = np.arange(len(autocorr))/FS  # Eje x para autocorrelación
                ax1 = plt.gca()
                l1, = ax1.plot(t, senal, color='blue', label='Señal')
                ax1.set_xlabel('Tiempo (s)')
                ax1.set_ylabel('Señal (azul)')
                ax2 = ax1.twinx()
                l2, = ax2.plot(lag, autocorr, color='orange', alpha=0.7, label='Autocorrelación')
                ax2.set_ylabel('Autocorrelación (naranja)')
                plt.title(f'Señal y Autocorrelación - {record}')
                # Leyenda combinada
                lines = [l1, l2]
                labels = [line.get_label() for line in lines]
                ax1.legend(lines, labels, loc='upper right')
                plt.tight_layout()
                # Guardar en la subcarpeta 'plots' con fecha
                record_filename = os.path.basename(record)
                plt.savefig(os.path.join(PLOTS_DIR, f'{record_filename}_{fecha_str}.png'))
                plt.close()
        except Exception as e:
            print(f"Error procesando {record}: {e}")
        pbar.update(1)  # Actualizar barra de progreso
end_total = time.time()

print(f"Tiempo total de procesamiento: {end_total-start_total:.2f} s")



# Crear DataFrame de resultados
df_out = pd.DataFrame(resultados, columns=['record', 'RR_intervalo', 'HR_bpm', 'mean_HR', 'std_HR'])
df_out.to_csv(os.path.join(DATA_DIR, 'resultados_ecg.csv'), index=False)
print("Procesamiento completado. Resultados guardados en 'resultados_ecg.csv'.")

# Graficar y guardar el scatter plot de HR media vs std para todos los puntos (sin filtro)
mean_hrs = df_out['mean_HR'].values
std_hrs = df_out['std_HR'].values
plt.figure(figsize=(8,6))
plt.scatter(mean_hrs, std_hrs, alpha=0.6)
plt.xlabel("Frecuencia cardíaca media (bpm)")
plt.ylabel("Variabilidad (desviación estándar, bpm)")
plt.title("Distribución de HR media vs variabilidad")
plt.tight_layout()
scatter_path = os.path.join(PLOTS_DIR, f'scatter_hr_std_{fecha_str}.png')
plt.savefig(scatter_path)
plt.close()
print(f"Scatter plot guardado en {scatter_path}")
