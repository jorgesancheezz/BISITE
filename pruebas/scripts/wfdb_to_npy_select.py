import os
import random
import numpy as np
import wfdb

# ===========================
# CONFIGURACIÓN
# ===========================
root_dir = r"C:\Users\BISITE-NEL\Desktop\pruebas\p10"
output_dir = r"C:\Users\BISITE-NEL\Desktop\pruebas\PULSOVITAL\npy_output"
selection_fraction = 0.30  # 30%
# reproducibility
random_seed = 42

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "AF"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "NSR"), exist_ok=True)

# ===========================
# FUNCIONES DE CLASIFICACIÓN
# ===========================

def is_NSR(aux_notes):
    """
    NSR según tu regla:
    Debe aparecer un texto que comience por '(N' y en algún punto aparezca ')'.
    """
    if not aux_notes:
        return False

    inside_nsr = False
    for note in aux_notes:
        if note and note.startswith("(N"):
            inside_nsr = True
        elif note == ")" and inside_nsr:
            return True

    return False


def is_AF(aux_notes):
    """
    AF según tu regla:
    Comienza '(AFIB' o '(AF' y termina con ')'.
    """
    if not aux_notes:
        return False

    inside_af = False
    for note in aux_notes:
        if note and (note.startswith("(AFIB") or note.startswith("(AF")):
            inside_af = True
        elif note == ")" and inside_af:
            return True

    return False


# ===========================
# CONVERTIR WFDB → NPY
# ===========================

def convert_to_npy(record_path, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    # Cargar señal
    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    signal = rec.p_signal
    ann_samples = np.array(ann.sample)
    ann_symbols = np.array(ann.symbol)
    aux_notes = np.array(ann.aux_note)

    # Guardar npy
    np.save(os.path.join(out_folder, "signal.npy"), signal)
    np.save(os.path.join(out_folder, "annotation_samples.npy"), ann_samples)
    np.save(os.path.join(out_folder, "annotation_symbols.npy"), ann_symbols)
    np.save(os.path.join(out_folder, "aux_notes.npy"), aux_notes)

    metadata = {
        "fs": rec.fs,
        "n_samples": signal.shape[0],
        "n_channels": signal.shape[1] if signal.ndim > 1 else 1,
        "sig_name": getattr(rec, 'sig_name', None),
        "baseline": getattr(rec, 'baseline', None),
        "units": getattr(rec, 'units', None)
    }
    np.save(os.path.join(out_folder, "metadata.npy"), metadata)

    print(f"[OK] Guardado: {out_folder}")


# ===========================
# BUSCAR TODOS LOS REGISTROS
# ===========================

records = []

for root, dirs, files in os.walk(root_dir):
    for f in files:
        if f.endswith(".hea"):
            record_path = os.path.join(root, f).replace(".hea", "")
            records.append(record_path)

print(f"Total registros detectados: {len(records)}")

# ===========================
# SELECCIONAR 30%
# ===========================

n_select = max(1, int(len(records) * selection_fraction))
random.seed(random_seed)
selected = random.sample(records, n_select)

print(f"Seleccionando aleatoriamente {n_select} registros (30%) con semilla {random_seed}.")


# ===========================
# PROCESAR REGISTROS
# ===========================

processed = []

for rec_path in selected:
    try:
        ann = wfdb.rdann(rec_path, "atr")
        aux_notes = ann.aux_note
    except Exception as e:
        print(f"[ERROR] leyendo anotaciones de {rec_path}: {e}")
        continue

    # Clasificar según tus reglas
    if is_AF(aux_notes):
        tag = "AF"
    elif is_NSR(aux_notes):
        tag = "NSR"
    else:
        print(f"[SKIP] {rec_path} no es AF ni NSR según tus reglas.")
        continue

    rec_name = os.path.basename(rec_path)
    out_folder = os.path.join(output_dir, tag, rec_name)

    try:
        convert_to_npy(rec_path, out_folder)
        processed.append((rec_path, tag, out_folder))
    except Exception as e:
        print(f"[ERROR] al convertir {rec_path}: {e}")

# guardar CSV con registros procesados
import csv
csv_path = os.path.join(output_dir, 'processed_records.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
    writer = csv.writer(cf)
    writer.writerow(['record', 'tag', 'out_folder'])
    for r,t,o in processed:
        writer.writerow([r, t, o])
    print(f"Processed records CSV saved to: {csv_path}")

print("=== PROCESO COMPLETO ===")
