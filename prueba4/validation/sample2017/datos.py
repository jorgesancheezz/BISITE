import os

# Carpeta donde guardaste los registros
data_path = "C:\\Users\\BISITE-NEL\\Desktop\\pruebas\\prueba4\\validation\\sample2017\\validation"

# Filtrar solo los archivos .hea (cabeceras)
records = [f.replace(".hea", "") for f in os.listdir(data_path) if f.endswith(".hea")]

print("Registros disponibles:", records)
