import wfdb

record_name = records[0]  # Elegimos el primero
record_path = os.path.join(data_path, record_name)

# Leer las señales y metadatos
record = wfdb.rdrecord(record_path)

# Leer anotaciones (si existen)
ann = wfdb.rdann(record_path, 'atr')

print("Nombre:", record.record_name)
print("Número de muestras:", record.sig_len)
print("Frecuencia de muestreo:", record.fs)
print("Señales disponibles:", record.sig_name)

# Mostrar primeras anotaciones
print("Anotaciones:", ann.sample[:10], ann.symbol[:10])
