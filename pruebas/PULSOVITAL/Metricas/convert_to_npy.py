import os
import numpy as np
import wfdb
import pandas as pd

# Rutas
source_folder = "p10/"
output_nsr = "p10/NSR_signals.npy"
output_af = "p10/AF_signals.npy"

# Listas para almacenar las señales
nsr_signals = []
af_signals = []

# Actualización de los símbolos para clasificar NSR y AF
# Basado en la inspección de los archivos .atr
NSR_SYMBOLS = ["N"]  # Símbolos que representan NSR
AF_SYMBOLS = ["A"]  # Símbolos que representan AF

# Longitud fija para las señales
FIXED_LENGTH = 3000

# Función para ajustar la longitud de las señales
def adjust_signal_length(signal, length):
    """
    Ajusta la longitud de una señal a un tamaño fijo mediante truncamiento o interpolación.
    """
    if len(signal) > length:
        return signal[:length]
    elif len(signal) < length:
        return np.interp(
            np.linspace(0, len(signal) - 1, length),
            np.arange(len(signal)),
            signal
        )
    return signal

# Función para clasificar señales según las anotaciones
def classify_signal(record_name, path):
    """
    Clasifica segmentos de una señal como NSR o AF según las anotaciones.
    Solo incluye segmentos que sean exclusivamente NSR o exclusivamente AF.
    """
    # Leer el registro y las anotaciones
    record = wfdb.rdrecord(os.path.join(path, record_name))
    annotation = wfdb.rdann(os.path.join(path, record_name), 'atr')

    # Extraer la señal
    signal = record.p_signal

    # Agregar depuración para verificar el procesamiento de señales
    print("Procesando registro:", record_name)
    print("Número de anotaciones:", len(annotation.sample))

    # Clasificar según las anotaciones
    for i in range(len(annotation.sample) - 1):
        start = annotation.sample[i]
        end = annotation.sample[i + 1]
        segment = signal[start:end]
        symbols = annotation.symbol[i:i + 1]

        # Depuración: Verificar cada segmento
        print(f"Segmento {i}: Longitud {len(segment)}, Símbolos {symbols}")

        # Verificar si el segmento es exclusivamente NSR o AF
        if len(segment) > 0:  # Asegurar que el segmento no esté vacío
            segment = adjust_signal_length(segment.flatten(), FIXED_LENGTH)  # Ajustar longitud
            segment = segment.reshape(-1, 1)  # Agregar dimensión adicional
            if all(sym in NSR_SYMBOLS for sym in symbols):
                nsr_signals.append(segment)
            elif all(sym in AF_SYMBOLS for sym in symbols):
                af_signals.append(segment)

# Depuración: Imprimir el número de señales clasificadas
print(f"Total señales NSR clasificadas: {len(nsr_signals)}")
print(f"Total señales AF clasificadas: {len(af_signals)}")

# Leer el archivo CSV para obtener los registros NSR y AF
csv_path = "pruebas/prueba6/resultados2/estadisticas_signales.csv"
data = pd.read_csv(csv_path)

# Filtrar señales exclusivamente NSR o AF
nsr_records = data[(data['is_nsr'] == True) & (data['is_af'] == False)]['record'].sample(100, random_state=42).tolist()
af_records = data[(data['is_af'] == True) & (data['is_nsr'] == False)]['record'].sample(100, random_state=42).tolist()

# Procesar solo los registros filtrados
for record_name in nsr_records:
    classify_signal(record_name, source_folder)

for record_name in af_records:
    classify_signal(record_name, source_folder)

# Leer el archivo chunks_af.csv para obtener segmentos AF
chunks_af_path = "pruebas/prueba6/resultados2/chunks_af.csv"
chunks_af = pd.read_csv(chunks_af_path)

# Procesar los segmentos AF del archivo chunks_af.csv
for _, row in chunks_af.iterrows():
    record_name = row['record']
    start_sample = int(row['chunk_start_sample'])
    end_sample = int(row['chunk_end_sample'])

    # Leer el registro y extraer el segmento
    record = wfdb.rdrecord(os.path.join(source_folder, record_name))
    signal = record.p_signal[start_sample:end_sample]

    # Ajustar la longitud del segmento y agregarlo a la lista AF
    if len(signal) > 0:
        segment = adjust_signal_length(signal.flatten(), FIXED_LENGTH)
        segment = segment.reshape(-1, 1)
        af_signals.append(segment)

# Convertir listas a arreglos numpy con la forma adecuada
if nsr_signals:
    nsr_signals = np.array(nsr_signals)
else:
    nsr_signals = np.empty((0, FIXED_LENGTH, 1))

if af_signals:
    af_signals = np.array(af_signals)
else:
    af_signals = np.empty((0, FIXED_LENGTH, 1))

# Guardar las señales en archivos .npy
np.save(output_nsr, nsr_signals)
np.save(output_af, af_signals)

print(f"Señales NSR guardadas en {output_nsr} con forma {nsr_signals.shape}")
print(f"Señales AF guardadas en {output_af} con forma {af_signals.shape}")

# Filtrar señales NSR y AF para limitar a 1024 ejemplos cada una
nsr_signals = nsr_signals[:1024]
af_signals = af_signals[:1024]

# Guardar las señales filtradas en nuevos archivos .npy
np.save("p10/NSR_signals_1024.npy", nsr_signals)
np.save("p10/AF_signals_1024.npy", af_signals)

print(f"Señales NSR limitadas a 1024 guardadas en p10/NSR_signals_1024.npy con forma {nsr_signals.shape}")
print(f"Señales AF limitadas a 1024 guardadas en p10/AF_signals_1024.npy con forma {af_signals.shape}")

# Cargar los archivos completos
nsr_signals = np.load("p10/NSR_signals.npy")
af_signals = np.load("p10/AF_signals.npy")

# Limitar a 1024 señales
nsr_signals_1024 = nsr_signals[:1024]
af_signals_1024 = af_signals[:1024]

# Guardar los archivos limitados
np.save("p10/NSR_signals_1024.npy", nsr_signals_1024)
np.save("p10/AF_signals_1024.npy", af_signals_1024)