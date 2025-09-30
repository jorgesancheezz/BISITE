import wfdb
import pandas as pd
import os
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'p10'))
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resultados2', 'estadisticas_signales.csv'))
df = pd.read_csv(csv_path)

is_af_list = []
for rec in tqdm(df['record'], desc="Buscando regiones AFIB"):
    rec_path = os.path.join(base_path, rec)
    rec_path = os.path.splitext(rec_path)[0]
    is_af = False
    try:
        ann = wfdb.rdann(rec_path, 'atr')
        aux = getattr(ann, 'aux_note', None)
        if aux is not None:
            inside_af = False
            for note in aux:
                if note and (note.startswith('(AFIB') or note.startswith('(AF')):
                    inside_af = True
                elif note and note == ')' and inside_af:
                    is_af = True
                    break
    except Exception:
        pass
    is_af_list.append(is_af)
df['is_af'] = is_af_list
df.to_csv(csv_path, index=False)
print("Columnas añadidas: 'is_af' en estadisticas_signales.csv (AFIB detectados si existen).")
