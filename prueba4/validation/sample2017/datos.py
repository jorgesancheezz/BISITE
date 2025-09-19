import os

# Carpeta donde guardaste los registros
data_path = "C:\\Users\\BISITE-NEL\\Desktop\\pruebas\\prueba4\\validation\\sample2017\\validation"

# Filtrar solo los archivos .hea (cabeceras)
records = [f.replace(".hea", "") for f in os.listdir(data_path) if f.endswith(".hea")]

print("Registros disponibles:", records)

import wfdb

record_name = records[0]  # Elegimos el primero
record_path = os.path.join(data_path, record_name)

# Leer las señales y metadatos
record = wfdb.rdrecord(record_path)

print("Nombre:", record.record_name)
print("Número de muestras:", record.sig_len)
print("Frecuencia de muestreo:", record.fs)
print("Señales disponibles:", record.sig_name)

import matplotlib.pyplot as plt

# Solo la primera señal
signal = record.p_signal[:,0]

plt.figure(figsize=(12,4))
plt.plot(signal[:2000])  # primeras 2000 muestras
plt.title(f"Señal {record.sig_name[0]} de {record.record_name}")
plt.xlabel("Muestra")
plt.ylabel("Amplitud")
plt.show()