# Librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Parámetros de la simulación
n_patients = 100          # Número de pacientes simulados
fs = 250                  # Frecuencia de muestreo del ECG (Hz)
duration_s = 30           # Duración del ECG (segundos)
n_samples = fs * duration_s  # Número de muestras por ECG

# Función para generar ECG simplificado
def generate_ecg_signal(is_af=False, n_samples=7500):
    """
    Genera un ECG simulado:
    - is_af: True simula fibrilación auricular con irregularidad
    - n_samples: número de puntos de la señal
    """
    t = np.linspace(0, 30, n_samples)
    # Señal base: latido regular (senoidal simplificado)
    signal = 0.5*np.sin(2*np.pi*1.0*t) + 0.05*np.random.randn(n_samples)
    if is_af:
        # Añadir irregularidad para simular FA
        jitter = np.random.normal(0, 0.1, n_samples)
        signal += jitter
    return signal.tolist()


# Función de filtrado bandpass (0.5-40 Hz)
def bandpass_filter(signal, fs, low=0.5, high=40):
    """
    Aplica filtro pasabanda para eliminar ruido muy bajo o alto.
    """
    b, a = butter(3, [low/(0.5*fs), high/(0.5*fs)], btype='band')
    return filtfilt(b, a, signal)

# Municipios simulados
municipalities = ['Zamora', 'Benavente', 'Toro', 'Morales', 'Villalpando']

# Lista para guardar registros
records = []

for pid in range(1, n_patients+1):
    age = np.random.randint(20, 90)                  # Edad aleatoria
    sex = np.random.choice(['M','F'])                # Sexo aleatorio
    municipality = np.random.choice(municipalities)  # Municipio aleatorio
    
    af_detected = np.random.choice([0,1], p=[0.9,0.1])  # 10% FA detectada
    # Simular confirmación médica con 10% de error
    af_confirmed = af_detected if np.random.rand() > 0.1 else 1-af_detected
    
    quality = np.random.choice(['Good','Noisy'], p=[0.85,0.15])  # Calidad de señal
    ecg_signal = generate_ecg_signal(is_af=(af_detected==1), n_samples=n_samples)
    
    records.append({
        'patient_id': pid,
        'age': age,
        'sex': sex,
        'municipality': municipality,
        'ecg_signal': ecg_signal,
        'quality': quality,
        'af_detected': af_detected,
        'af_confirmed': af_confirmed
    })

# Convertir a DataFrame
df_sim = pd.DataFrame(records)

# Guardar CSV
df_sim.to_csv("simulated_pulso_vital.csv", index=False)
print("Base de datos simulada creada: simulated_pulso_vital.csv")


# Crear carpeta de resultados
os.makedirs('resultados', exist_ok=True)

# Guardar primeras filas
df_sim.head().to_csv('resultados/primeras_filas.csv', index=False)

# Distribución de edad
plt.figure(figsize=(8,4))
sns.histplot(df_sim['age'], bins=15, kde=True)
plt.title("Distribución de edades")
plt.xlabel("Edad")
plt.ylabel("Cantidad de pacientes")
plt.tight_layout()
plt.savefig('resultados/edad_hist.png')
plt.close()

# Proporción FA detectada vs confirmada
plt.figure(figsize=(6,4))
sns.countplot(data=df_sim, x='af_detected', label='Detectada')
sns.countplot(data=df_sim, x='af_confirmed', label='Confirmada', color='red', alpha=0.5)
plt.title("FA detectada vs confirmada")
plt.xlabel("FA (0=No, 1=Sí)")
plt.ylabel("Cantidad de pacientes")
plt.legend(['Detectada','Confirmada'])
plt.tight_layout()
plt.savefig('resultados/fa_detectada_vs_confirmada.png')
plt.close()

# Cantidad por municipio
plt.figure(figsize=(8,4))
sns.countplot(data=df_sim, x='municipality')
plt.title("Número de registros por municipio")
plt.tight_layout()
plt.savefig('resultados/municipios.png')
plt.close()

# Tomamos un paciente con buena señal
example = df_sim[df_sim['quality']=='Good'].iloc[0]
signal_raw = np.array(example['ecg_signal'])

# Filtrado
signal_filtered = bandpass_filter(signal_raw, fs)

# Graficar ECG
plt.figure(figsize=(12,4))
plt.plot(signal_filtered[:1000])  # mostrar primeros 1000 puntos (~4s)
plt.title(f"ECG filtrado paciente {example['patient_id']} (FA detectada: {example['af_detected']})")
plt.xlabel("Muestras")
plt.ylabel("Voltaje (mV)")
plt.tight_layout()
plt.savefig('resultados/ecg_filtrado_paciente.png')
plt.close()

# Comparar af_detected vs af_confirmed
y_true = df_sim['af_confirmed']
y_pred = df_sim['af_detected']

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
np.savetxt('resultados/matriz_confusion.txt', cm, fmt='%d', header='Matriz de confusión')

# Reporte completo
report = classification_report(y_true, y_pred, digits=4)
with open('resultados/reporte_clasificacion.txt', 'w') as f:
    f.write(report)
