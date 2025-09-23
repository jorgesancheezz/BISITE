import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk
# Carpeta donde guardaste los registros
data_path = "C:\\Users\\BISITE-NEL\\Desktop\\pruebas\\prueba4\\validation\\sample2017\\validation"
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "resultados")  # carpeta dentro de prueba4
output_folder = os.path.abspath(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Lista de registros
records = [f.replace(".mat","") for f in os.listdir(data_path) if f.endswith(".mat")]

# DataFrame para guardar estadísticas
eda_df = pd.DataFrame(columns=[
    "record", "n_signals", "length_samples", "fs", "duration_sec",
    "mean_amplitude", "std_amplitude", "hr_mean", "hr_std"
])

# Recorrer registros
for rec in records:  # Limitar a los primeros 10 para prueba
    try:
        r = wfdb.rdrecord(os.path.join(data_path, rec))
        if r.p_signal is None:
            print(f"Registro {rec} vacío, se omite")
            continue

        signals = r.p_signal
        n_signals = r.n_sig
        length_samples = r.sig_len
        fs = r.fs
        duration_sec = length_samples / fs
        mean_amp = np.mean(signals)
        std_amp = np.std(signals)

        # Usamos la primera señal para análisis y gráficos
        signal = signals[:,0]
        
        # Detectar picos R con NeuroKit2
        ecg_cleaned = nk.ecg_clean(signal, sampling_rate=fs)
        ecg_peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
        heart_rate = nk.ecg_rate(ecg_peaks, sampling_rate=fs)
        hr_mean = heart_rate.mean() if len(heart_rate) > 0 else np.nan
        hr_std = heart_rate.std() if len(heart_rate) > 0 else np.nan

        # Guardar métricas en DataFrame
        new_row = pd.DataFrame([{
            "record": rec,
            "n_signals": n_signals,
            "length_samples": length_samples,
            "fs": fs,
            "duration_sec": duration_sec,
            "mean_amplitude": mean_amp,
            "std_amplitude": std_amp,
            "hr_mean": hr_mean,
            "hr_std": hr_std
        }])
        eda_df = pd.concat([eda_df, new_row], ignore_index=True)

        # Graficar primera señal con picos R (primeras 6000 muestras)
        plt.figure(figsize=(12,4))
        plt.plot(ecg_cleaned[:6000], label="ECG")
        r_peaks = info["ECG_R_Peaks"]
        r_peaks_in_window = r_peaks[r_peaks < 6000]  # solo las que están en las primeras 6000 muestras
        plt.scatter(r_peaks_in_window,ecg_cleaned[r_peaks_in_window], color='red', marker='x', label='R Peaks')
        plt.title(f"Registro {rec} - primeras 6000 muestras")
        plt.xlabel("Muestra")
        plt.ylabel("Amplitud")
        plt.legend()

        # Guardar la figura
        plt.savefig(os.path.join(output_folder, f"{rec}.png"))
        plt.close()

    except Exception as e:
        print(f"No se pudo procesar {rec}: {e}")

# Guardar CSV con todas las métricas
eda_df.to_csv("metrics.csv", index=False)
print("EDA completado. CSV y gráficos guardados.")
