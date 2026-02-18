import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score

# File paths
reference_files = [
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/003.npy",
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/004.npy",
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/005.npy",
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/006.npy",
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/007.npy",
]
synthetic_files = [
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/synthetic_scale_0.1270_noise_0.05.npz",
    "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/synthetic_scale_0.1290_noise_0.1.npz",
]

# Load data
psd_data = {}
for file in reference_files:
    key = file.split("/")[-1]
    data = np.load(file)
    if data.ndim > 1:
        psd_data[key] = np.mean(data, axis=0).flatten()  # Promediar si es multidimensional
    else:
        psd_data[key] = data

for file in synthetic_files:
    key = file.split("/")[-1]
    data = np.load(file)["data"]
    psd_data[key] = np.mean(data, axis=(0, 1)).flatten()  # Promediar correctamente las dimensiones

# Plot PSD comparison
plt.figure(figsize=(10, 6))
for key, psd in psd_data.items():
    plt.plot(psd, label=key)

plt.title("PSD Comparison: References vs Synthetic")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB/Hz)")
plt.legend()
plt.grid()
plt.tight_layout()

# Save plot
output_path = "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/psd_comparison.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
plt.show()

def calculate_metrics(y_true, y_pred):
    """Calcula sensibilidad y especificidad a partir de etiquetas verdaderas y predicciones."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensibilidad = tp / (tp + fn)
    especificidad = tn / (tn + fp)

    return sensibilidad, especificidad

# Ejemplo de uso
if __name__ == "__main__":
    # Etiquetas verdaderas y predicciones (ejemplo)
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]

    # Verificar matriz de confusión
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred))

    sensibilidad, especificidad = calculate_metrics(y_true, y_pred)

    print(f"Sensibilidad: {sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")