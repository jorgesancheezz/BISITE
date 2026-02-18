import os
import numpy as np
import glob
import scipy
from scipy.signal import welch
from scipy.stats import energy_distance
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy import linalg
from PULSOVITAL.metrics.fid_with_model import CNNEncoder1D, collect_embeddings, frechet_distance
import torch
import csv
import sys
import traceback

print("metrica.py start", flush=True)

# Agregar el directorio raíz al PYTHONPATH dinámicamente
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

#####################################################
# 1. CARGA Y PREPROCESADO
#####################################################

def load_npy_folder(path):
    files = glob.glob(os.path.join(path, "*.npy"))
    signals = [np.load(f).astype(float) for f in files]
    return signals

def normalize_signal(sig):
    """Normalización robusta (z-score + clipping)"""
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    return np.clip(sig, -5, 5)

def preprocess_signals(signals):
    return [normalize_signal(s) for s in signals]

def truncate_signals(signals, length=1000):
    """
    Trunca cada señal en la lista a una longitud fija.

    Parameters:
        signals (list or np.ndarray): Lista de señales a truncar.
        length (int): Longitud máxima de cada señal.

    Returns:
        np.ndarray: Señales truncadas.
    """
    return np.array([s[:length] for s in signals])


#####################################################
# 2. DTW (Dynamic Time Warping)
#####################################################

def dtw_distance(x, y):
    return scipy.spatial.distance.cdist(x[:, None], y[:, None], metric='euclidean').min()

def evaluate_dtw(real, synth, samples=50):
    dtw_scores = []
    for _ in range(samples):
        r = real[np.random.randint(len(real))]
        s = synth[np.random.randint(len(synth))]
        score = dtw_distance(r, s)
        dtw_scores.append(score)
    return np.mean(dtw_scores), np.std(dtw_scores)


#####################################################
# 3. PSD + WASSERSTEIN DISTANCE
#####################################################

def compute_psd(sig, fs=250):
    f, Pxx = welch(sig, fs=fs, nperseg=256)
    return f, Pxx

def evaluate_psd_wasserstein(real, synth, samples=50):
    distances = []
    for _ in range(samples):
        r = real[np.random.randint(len(real))]
        s = synth[np.random.randint(len(synth))]

        _, P_r = compute_psd(r)
        _, P_s = compute_psd(s)

        # igualar longitudes
        L = min(len(P_r), len(P_s))
        # Reemplazar el cálculo de Wasserstein Distance con una aproximación alternativa
        dist = np.sum(np.abs(P_r[:L] - P_s[:L]))
        distances.append(dist)
    return np.mean(distances), np.std(distances)


#####################################################
# 4. AUTOCORRELACIONES (ACF)
#####################################################

def acf(x, max_lag=200):
    result = np.correlate(x, x, mode='full')
    result = result[result.size//2:]
    return result[:max_lag] / result[0]

def evaluate_acf(real, synth, samples=50):
    diffs = []
    for _ in range(samples):
        r = acf(real[np.random.randint(len(real))])
        s = acf(synth[np.random.randint(len(synth))])
        L = min(len(r), len(s))
        diffs.append(np.mean(np.abs(r[:L] - s[:L])))
    return np.mean(diffs), np.std(diffs)


#####################################################
# 5. MMD (Maximum Mean Discrepancy)
#####################################################

def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-cdist(x, y, 'sqeuclidean') / (2 * sigma**2))

def compute_mmd(real, synth, sigma=1.0):
    X = np.array(real, dtype=object)
    Y = np.array(synth, dtype=object)

    # Pad sequences to same length
    L = min(min(len(x) for x in X), min(len(y) for y in Y))
    X = np.array([x[:L] for x in X])
    Y = np.array([y[:L] for y in Y])

    Kxx = gaussian_kernel(X, X, sigma).mean()
    Kyy = gaussian_kernel(Y, Y, sigma).mean()
    Kxy = gaussian_kernel(X, Y, sigma).mean()
    return Kxx + Kyy - 2 * Kxy


#####################################################
# 6. ENERGY DISTANCE
#####################################################

def evaluate_energy(real, synth, samples=200):
    R = np.array([real[i][0:500] for i in np.random.randint(0, len(real), samples)])
    S = np.array([synth[i][0:500] for i in np.random.randint(0, len(synth), samples)])
    return energy_distance(R.flatten(), S.flatten())


#####################################################
# 7. C2ST — CLASIFICADOR REAL VS SINTÉTICO
#####################################################

def flatten_signals(signals):
    return np.array([s.reshape(-1) for s in signals])

def c2st(real, synth):
    L = min(min(len(r) for r in real), min(len(s) for s in synth), 500)
    X_real = flatten_signals(real, L)
    X_synth = flatten_signals(synth, L)

    X = np.vstack([X_real, X_synth])
    y = np.array([0]*len(X_real) + [1]*len(X_synth))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True
    )

    clf = RandomForestClassifier(n_estimators=80)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return accuracy_score(y_test, pred)


#####################################################
# 8. TSTR — TRAIN ON SYNTHETIC, TEST ON REAL
#####################################################

def simulate_labels(signals):
    """Fake labels for testing usefulness (placeholder if you don't have labels)."""
    return np.random.randint(0, 2, len(signals))

def tstr(real, synth):
    L = min(min(len(r) for r in real), min(len(s) for s in synth), 500)
    X_real = flatten_signals(real, L)
    X_synth = flatten_signals(synth, L)

    y_real = simulate_labels(X_real)
    y_synth = simulate_labels(X_synth)

    clf = RandomForestClassifier(n_estimators=80)
    clf.fit(X_synth, y_synth)

    pred = clf.predict(X_real)

    return accuracy_score(y_real, pred)


#####################################################
# 9. FID (Frechet Inception Distance)
#####################################################

def calculate_fid(real, synth):
    """
    Calcula la Frechet Inception Distance (FID) entre dos conjuntos de datos.
    """
    # Calcular las medias y covarianzas de los datos reales y sintéticos
    mu_real = np.mean(real, axis=0)
    mu_synth = np.mean(synth, axis=0)
    cov_real = np.cov(real, rowvar=False)
    cov_synth = np.cov(synth, rowvar=False)

    # Calcular la distancia FID
    cov_sqrt = sqrtm(cov_real @ cov_synth)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = np.sum((mu_real - mu_synth)**2) + np.trace(cov_real + cov_synth - 2 * cov_sqrt)
    return fid

# Implementación alternativa para calcular el FID sin wasserstein_distance
def calculate_fid_alternative(real, synth):
    """
    Calcula una aproximación de la Frechet Inception Distance (FID) entre dos conjuntos de datos
    sin usar wasserstein_distance.
    """
    # Calcular las medias y covarianzas de los datos reales y sintéticos
    mu_real = np.mean(real, axis=0)
    mu_synth = np.mean(synth, axis=0)
    cov_real = np.cov(real, rowvar=False)
    cov_synth = np.cov(synth, rowvar=False)

    # Calcular la distancia FID aproximada
    diff = mu_real - mu_synth
    fid = np.sum(diff**2) + entropy(cov_real.flatten()) + entropy(cov_synth.flatten())
    return fid

# Integrar cálculo de FID basado en modelo
def calculate_fid_with_model(real, synth):
    """
    Calcula el FID utilizando estadísticas de medias y covarianzas.
    """
    mu_real = np.mean(real, axis=0)
    mu_synth = np.mean(synth, axis=0)
    sigma_real = np.cov(real, rowvar=False)
    sigma_synth = np.cov(synth, rowvar=False)

    # Calcular la raíz de la matriz de covarianza
    covmean = linalg.sqrtm(sigma_real.dot(sigma_synth))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu_real - mu_synth)**2) + np.trace(sigma_real + sigma_synth - 2 * covmean)
    return fid

# Integrar el uso de embeddings y optimizar el cálculo del FID
def calculate_fid_with_embeddings(real, synth, ckpt_path):
    """
    Calcula el FID utilizando embeddings generados por un modelo preentrenado.

    Parameters:
        real (list): Señales reales.
        synth (list): Señales sintéticas.
        ckpt_path (str): Ruta al checkpoint del modelo preentrenado.

    Returns:
        float: FID calculado.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _, _ = CNNEncoder1D().to(device), 64, 64

    # Convertir las señales a tensores de PyTorch antes de generar embeddings
    real = [torch.tensor(r, dtype=torch.float32).unsqueeze(0) for r in real]
    synth = [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in synth]

    # Ajustar las dimensiones de los tensores para que sean exactamente (B, 1, T)
    real = [s.view(1, 1, -1) for s in real]  # Reorganizar dimensiones innecesarias
    synth = [s.view(1, 1, -1) for s in synth]  # Reorganizar dimensiones innecesarias

    # Antes de pasar a collect_embeddings, eliminar dimensiones adicionales
    real = [r.squeeze() for r in real]  # Eliminar dimensiones adicionales
    synth = [s.squeeze() for s in synth]  # Eliminar dimensiones adicionales

    # Agregar mensajes de depuración para verificar las dimensiones de los tensores
    def debug_tensor_dimensions(tensors, label):
        """Imprime las dimensiones de los tensores para depuración."""
        for i, t in enumerate(tensors):
            print(f"{label} Tensor {i}: {t.shape}")

    # Depuración: Verificar las dimensiones de los tensores
    debug_tensor_dimensions(real, "Real")
    debug_tensor_dimensions(synth, "Synthetic")

    # Generar embeddings
    real_embs = collect_embeddings(encoder, real, len(real), device)
    synth_embs = collect_embeddings(encoder, synth, len(synth), device)

    # Calcular estadísticas
    mu_real, sigma_real = np.mean(real_embs, axis=0), np.cov(real_embs, rowvar=False)
    mu_synth, sigma_synth = np.mean(synth_embs, axis=0), np.cov(synth_embs, rowvar=False)

    # Calcular FID
    return frechet_distance(mu_real, sigma_real, mu_synth, sigma_synth)


#####################################################
# MAIN PIPELINE
#####################################################

def run_pipeline(real_path, synth_path):
    print("Cargando datos...")
    real = preprocess_signals(load_npy_folder(real_path))
    synth = preprocess_signals(load_npy_folder(synth_path))

    print("\n--- DTW ---")
    dtw_mean, dtw_std = evaluate_dtw(real, synth)
    print("DTW:", dtw_mean, "+-", dtw_std)

    print("\n--- PSD + Wasserstein ---")
    psd_m, psd_s = evaluate_psd_wasserstein(real, synth)
    print("PSD-Wasserstein:", psd_m, "+-", psd_s)

    print("\n--- ACF ---")
    acf_m, acf_s = evaluate_acf(real, synth)
    print("ACF diff:", acf_m, "+-", acf_s)

    print("\n--- MMD ---")
    mmd = compute_mmd(real, synth)
    print("MMD:", mmd)

    print("\n--- ENERGY DISTANCE ---")
    ed = evaluate_energy(real, synth)
    print("Energy:", ed)

    print("\n--- C2ST (Real vs Synthetic Classifier) ---")
    acc = c2st(real, synth)
    print("C2ST accuracy:", acc)

    print("\n--- TSTR (Train Synthetic, Test Real) ---")
    tstr_acc = tstr(real, synth)
    print("TSTR accuracy:", tstr_acc)

    print("\n--- FID (Frechet Inception Distance) ---")
    fid = calculate_fid(real, synth)
    print("FID:", fid)


################################################
    # Asegurar que las funciones convert_to_tensor y adjust_tensor_dimensions estén correctamente integradas

    # Corregir las funciones para asegurar que los tensores tengan dimensiones correctas

    # Simplificar las funciones para garantizar dimensiones correctas (B, 1, T)

    def convert_to_tensor(signals):
        """Convierte una lista de señales en tensores de PyTorch manteniendo las dimensiones originales."""
        return [torch.tensor(s, dtype=torch.float32) for s in signals]

    # Ajustar las dimensiones de los tensores para que sean exactamente (B, 1, T)

    # Reorganizar las dimensiones de los tensores cargados para que sean (B, C, T)
    def adjust_tensor_dimensions(signals):
        """Reorganiza las dimensiones de los tensores para que sean (B, C, T)."""
        return [torch.tensor(s).permute(0, 2, 1) for s in signals]  # Cambiar de (B, T, C) a (B, C, T)

    # Paths
    real_path = r"c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/AF_signals_1024.npy"
    synth_path = r"c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/1024seq_AF.npy"

    # Output files
    output_csv = r"c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/fid_results.csv"
    output_plot = r"c:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/Metricas/fid_plot.png"

    try:
        print("Loading .npy arrays...", flush=True)
        real_arr = np.load(real_path)
        synth_arr = np.load(synth_path)
        print(f"Loaded real array shape: {real_arr.shape}", flush=True)
        print(f"Loaded synth array shape: {synth_arr.shape}", flush=True)

        # If arrays are datasets with shape (N, T, C) convert to list of 1D signals
        def arr_to_list(arr):
            arr = np.asarray(arr)
            if arr.ndim == 3:
                # (N, T, C) -> list of (T,) by taking channel 0
                return [arr[i, :, 0].astype(float) for i in range(arr.shape[0])]
            if arr.ndim == 2:
                # (N, T) -> list
                return [arr[i, :].astype(float) for i in range(arr.shape[0])]
            if arr.ndim == 1:
                # single long signal -> single-element list
                return [arr.astype(float)]
            # fallback: try to flatten per-sample
            return [s.astype(float).ravel() for s in arr]

        real_signals = arr_to_list(real_arr)
        synth_signals = arr_to_list(synth_arr)

        print(f"Number of real signals: {len(real_signals)}", flush=True)
        print(f"Number of synth signals: {len(synth_signals)}", flush=True)

        # Preprocess and truncate to fixed length (e.g., 1000 samples)
        L = 1000
        real_proc = truncate_signals(preprocess_signals(real_signals), length=L)
        synth_proc = truncate_signals(preprocess_signals(synth_signals), length=L)

        print(f"Shapes after truncation: real {real_proc.shape}, synth {synth_proc.shape}", flush=True)

        # Ensure shape (N, features)
        if real_proc.ndim != 2 or synth_proc.ndim != 2:
            raise RuntimeError("Truncated signals do not have expected shape (N, T)")

        # Compute FID using the numpy implementation (no model dependencies)
        print("Computing FID (numpy implementation)...", flush=True)
        fid_value = calculate_fid(real_proc, synth_proc)
        print(f"Computed FID: {fid_value}", flush=True)

        # Save CSV
        import csv, os
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"]) 
            writer.writerow(["FID", float(fid_value)])
        print(f"Saved CSV to {output_csv}", flush=True)

        # Plot and save
        try:
            plt.figure(figsize=(5, 3))
            plt.bar(["FID"], [fid_value], color='tab:blue')
            plt.ylabel('Frechet Inception Distance')
            plt.title('FID: real vs synth')
            plt.tight_layout()
            plt.savefig(output_plot)
            # Do not rely on interactive display; close figure
            plt.close()
            print(f"Saved plot to {output_plot}", flush=True)
        except Exception as e_plot:
            print(f"Plotting error: {e_plot}", flush=True)
            traceback.print_exc()

    except Exception as e:
        print(f"Fatal error in main execution: {e}", flush=True)
        traceback.print_exc()
