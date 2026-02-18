import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def plot_mean_std(data: np.ndarray, label: str, color: str, ax: plt.Axes):
    mu = np.mean(data, axis=0).flatten()
    sd = np.std(data, axis=0).flatten()
    ax.plot(mu, label=f"{label} (mean)", color=color)
    ax.fill_between(range(len(mu)), mu - sd, mu + sd, color=color, alpha=0.2, label=f"{label} (std)")

def main():
    # File paths
    reference_files = [
        "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/003.npy",
        "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/004.npy",
        "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/005.npy",
        "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/006.npy",
        "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/007.npy",
    ]
    synthetic_files = [
        "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/synthetic_scale_0.1270_noise_0.05.npz",
        "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/synthetic_scale_0.1290_noise_0.1.npz",
    ]

    # Load data
    reference_data = [np.load(file) for file in reference_files]
    synthetic_data = [np.load(file)["data"] for file in synthetic_files]

    # Prepare plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_title("Comparación de PSD: Referencias y Sintéticos seleccionados")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Potencia (dB/Hz)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, 120)
    ax.set_ylim(-80, -20)

    # Ensure data is processed correctly
    reference_data = [np.mean(np.load(file), axis=0) for file in reference_files]
    synthetic_data = [np.mean(np.load(file)["data"], axis=0) for file in synthetic_files]

    # Plot reference data
    for i, data in enumerate(reference_data):
        ax.plot(data, label=f"Reference {i+1}", color=f"C{i}")

    # Plot synthetic data
    for i, data in enumerate(synthetic_data):
        ax.plot(data, label=f"Synthetic {i+1}", color=f"C{i+5}")

    # Ensure labels are correctly assigned and visible
    for i, data in enumerate(reference_data):
        ax.plot(range(len(data)), data, label=f"Reference {i+1}", color=f"C{i}")

    for i, data in enumerate(synthetic_data):
        ax.plot(range(len(data)), data, label=f"Synthetic {i+1}", color=f"C{i+5}")

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Save plot
    output_path = "C:/Users/BISITE-NEL/Desktop/pruebas/PULSOVITAL/results/psd_comparison_mean_std.png"
    ensure_dir(output_path)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()