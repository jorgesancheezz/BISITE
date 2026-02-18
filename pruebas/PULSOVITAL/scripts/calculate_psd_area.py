import numpy as np
from scipy.signal import welch
import csv

# Archivos de referencia y sintéticos
files = [
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/sintetico.npy",
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/003.npy",
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/004.npy",
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/005.npy",
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/006.npy",
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/007.npy"
]

# Frecuencia de muestreo
fs = 250.0

# Cargar el archivo de referencia
ref_data = np.load(files[0])
ref_freqs, ref_psd = welch(ref_data[:, :, 0].mean(axis=0), fs=fs, nperseg=256, noverlap=128)

# Inicializar resultados como lista
results = []

# Iterar sobre los archivos de referencia
for file in files[1:]:
    synth_data = np.load(file)
    synth_freqs, synth_psd = welch(synth_data[:, :, 0].mean(axis=0), fs=fs, nperseg=256, noverlap=128)

    # Interpolación para frecuencias comunes
    common_freqs = np.intersect1d(ref_freqs, synth_freqs)
    ref_interp = np.interp(common_freqs, ref_freqs, ref_psd)
    synth_interp = np.interp(common_freqs, synth_freqs, synth_psd)

    # Calcular área de diferencia
    area_difference = np.trapz(np.abs(ref_interp - synth_interp), common_freqs)
    results.append(area_difference)

# Guardar resultados en un archivo CSV
output_csv = "PULSOVITAL/results/psd_area_differences.csv"
with open(output_csv, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    # Escribir encabezados
    csvwriter.writerow(["Synthetic", "003.npy", "004.npy", "005.npy", "006.npy", "007.npy"])
    # Escribir fila con los resultados
    csvwriter.writerow(["sintetico.npy"] + results)

print(f"Cálculo completado. Resultados guardados en {output_csv}")