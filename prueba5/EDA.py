import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración de rutas ---
data_path = "C:\\Users\\BISITE-NEL\\Desktop\\pruebas\\prueba5\\lung_cancer_dataset.csv"
output_folder = "resultados_frequencies"
os.makedirs(output_folder, exist_ok=True)

# --- Leer CSV ---
df = pd.read_csv(data_path)

# --- Codificar variables categóricas binarias ---
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['family_history'] = df['family_history'].map({'No':0, 'Yes':1})
df['copd_diagnosis'] = df['copd_diagnosis'].map({'No':0, 'Yes':1})
df['asbestos_exposure'] = df['asbestos_exposure'].map({'No':0, 'Yes':1})
df['secondhand_smoke_exposure'] = df['secondhand_smoke_exposure'].map({'No':0, 'Yes':1})
df['lung_cancer'] = df['lung_cancer'].map({'No':0, 'Yes':1})

# --- Codificar alcohol_consumption ---
# None=0, Moderate=1, Heavy=2
df['alcohol_consumption'] = df['alcohol_consumption'].map({'None':0, 'Moderate':1, 'Heavy':2})

# --- Variables categóricas a analizar ---
categorical_vars = ['gender','copd_diagnosis','alcohol_consumption',
                    'family_history','asbestos_exposure','secondhand_smoke_exposure']

# --- Generar tablas y gráficos ---
for col in categorical_vars:
    # Tabla de frecuencias
    freq_table = df[col].value_counts().reset_index()
    freq_table.columns = [col, 'count']
    freq_table['percentage'] = (freq_table['count'] / freq_table['count'].sum() * 100).round(2)
    
    # Distribución según lung_cancer
    lung_table = pd.crosstab(df[col], df['lung_cancer'], normalize='index') * 100
    lung_table = lung_table.rename(columns={0:'No Cancer %', 1:'Cancer %'}).round(2)
    
    # Combinar tablas
    freq_table = freq_table.merge(lung_table, left_on=col, right_index=True)
    
    # Mostrar tabla por pantalla
    print(f"\nTabla de frecuencias para {col}:")
    print(freq_table)
    
    # Guardar CSV
    csv_path = os.path.join(output_folder, f"{col}_frequency_table.csv")
    freq_table.to_csv(csv_path, index=False)
    
    # --- Gráfico de barras simple (proporción total) ---
    plt.figure(figsize=(5,4))
    sns.barplot(x=freq_table[col], y=freq_table['percentage'], palette='Set2')
    plt.ylabel("Porcentaje total (%)")
    plt.title(f"{col} - Porcentaje total")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{col}_bar_total.png"))
    plt.close()
    
    # --- Stacked bar plot (proporción de lung_cancer por categoría) ---
    plt.figure(figsize=(5,4))
    lung_table.plot(kind='bar', stacked=True, colormap='coolwarm', width=0.6)
    plt.ylabel("Porcentaje dentro de la categoría (%)")
    plt.title(f"{col} - Distribución de Lung Cancer")
    plt.legend(title='Lung Cancer', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{col}_stacked_lung_cancer.png"))
    plt.close()

print(f"\nTodas las tablas y gráficos se han guardado en '{output_folder}'.")
