import wfdb
import pandas as pd
import os
from tqdm import tqdm

# Ruta a la base de datos y estadisticas
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'p10'))
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resultados2', 'estadisticas_signales.csv'))
df = pd.read_csv(csv_path)

# Añadir columna con el número de latidos normales ('N')
num_N_list = []
for rec in tqdm(df['record'], desc="Contando latidos normales ('N')"):
    rec_path = os.path.join(base_path, rec)
    rec_path = os.path.splitext(rec_path)[0]
    num_N = 0
    try:
        ann = wfdb.rdann(rec_path, 'atr')
        symbols = getattr(ann, 'symbol', [])
        num_N = sum(1 for s in symbols if s == 'N')
    except Exception as e:
        pass
    num_N_list.append(num_N)
df['num_N'] = num_N_list
print(f"Total de latidos normales (N) en todos los registros: {df['num_N'].sum()}")
df.to_csv(csv_path, index=False)
print("Columna 'num_N' añadida a estadisticas_signales.csv.")

