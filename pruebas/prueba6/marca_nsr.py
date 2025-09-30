import wfdb
import pandas as pd
import os
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'p10'))
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resultados2', 'estadisticas_signales.csv'))
df = pd.read_csv(csv_path)

is_nsr_list = []
for rec in tqdm(df['record'], desc="Buscando regiones NSR"):
    rec_path = os.path.join(base_path, rec)
    rec_path = os.path.splitext(rec_path)[0]
    is_nsr = False
    try:
        ann = wfdb.rdann(rec_path, 'atr')
        aux = getattr(ann, 'aux_note', None)
        if aux is not None:
            inside_nsr = False
            for note in aux:
                if note and note.startswith('(N'):
                    inside_nsr = True
                elif note and note == ')' and inside_nsr:
                    is_nsr = True
                    break
    except Exception:
        pass
    is_nsr_list.append(is_nsr)
df['is_nsr'] = is_nsr_list
df.to_csv(csv_path, index=False)
print("Columna 'is_nsr' añadida a estadisticas_signales.csv (solo registros con regiones NSR).")
