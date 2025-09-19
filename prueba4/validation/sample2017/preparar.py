import neurokit2 as nk
import wfdb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

data_path = "C:\\Users\\BISITE-NEL\\Desktop\\pruebas\\prueba4\\validation\\sample2017\\validation"
output_folder = "resultados2"
max_len = 5000
os.makedirs(output_folder, exist_ok=True)
list_of_signals_fixed = []
eda_df = pd.DataFrame()

# Obtener nombres de registros (sin extensión)
records = [f.split(".")[0] for f in os.listdir(data_path) if f.endswith(".mat")]
for rec in records:
    r = wfdb.rdrecord(os.path.join(data_path, rec))
    signal = r.p_signal[:,0]  # primer canal

    # Limpiar ECG
    clean_signal = nk.ecg_clean(signal, sampling_rate=r.fs)
    signal_norm = (clean_signal - np.mean(clean_signal)) / np.std(clean_signal)

    if len(signal_norm) > max_len:
        signal_fixed = signal_norm[:max_len]
    else:
        signal_fixed = np.pad(signal_norm, (0, max_len - len(signal_norm)), 'constant')

    list_of_signals_fixed.append(signal_fixed)

    # Guardar estadísticas en DataFrame
    new_row = pd.DataFrame([{
        "record": rec,
        "n_signals": r.n_sig,
        "length_samples": r.sig_len,
        "fs": r.fs,
        "duration_sec": r.sig_len / r.fs,
        "mean_amplitude": np.mean(signal),
        "std_amplitude": np.std(signal)
    }])
    eda_df = pd.concat([eda_df, new_row], ignore_index=True)
signals_array = np.array(list_of_signals_fixed)
print("Array de señales con forma:", signals_array.shape)
eda_df.to_csv(os.path.join(output_folder, "estadisticas_signales.csv"), index=False)
print("Estadísticas guardadas en 'estadisticas_signales.csv'")
print(eda_df)
# Guardar tensor para ML
np.save(os.path.join(output_folder, "X_ml.npy"), signals_array)
signals_matrix = np.array(list_of_signals_fixed)
corr_matrix = np.corrcoef(signals_matrix)  # shape = (n_samples, n_samples)

plt.figure(figsize=(10,8))
plt.imshow(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(label="Correlación")
plt.title("Matriz de correlación entre registros")#no es interesante
plt.show()