import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
import logging
from datetime import datetime

# Crear carpeta para guardar resultados dentro de prueba5
output_folder = os.path.join(os.path.dirname(__file__), "resultados_modelo_smote")
os.makedirs(output_folder, exist_ok=True)

# Utilidad para nombre único de imagen
def unique_fig_name(base):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_folder, f"{base}_{timestamp}.png")

# Configurar logging sin sobreescribir
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_folder, f"smote_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logging.info("Inicio del script SMOTE")

# Cargar y preparar los datos
data_path = os.path.join(os.path.dirname(__file__), "lung_cancer_dataset.csv")
df = pd.read_csv(data_path)

logging.info(f"Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")

if df['alcohol_consumption'].isnull().sum() > 0:
    df['alcohol_consumption'] = df['alcohol_consumption'].fillna('Unknown')
    
df['gender'] = df['gender'].map({'Male':0,'Female':1})
df['family_history'] = df['family_history'].map({'No':0,'Yes':1})
df['copd_diagnosis'] = df['copd_diagnosis'].map({'No':0,'Yes':1})
df['asbestos_exposure'] = df['asbestos_exposure'].map({'No':0,'Yes':1})
df['secondhand_smoke_exposure'] = df['secondhand_smoke_exposure'].map({'No':0,'Yes':1})
df['lung_cancer'] = df['lung_cancer'].map({'No':0,'Yes':1})
df['alcohol_consumption'] = df['alcohol_consumption'].map({'None':0,'Moderate':1,'Heavy':2})
df['radon_exposure'] = df['radon_exposure'].map({'Low':0,'Medium':1,'High':2})

# Crear risk_score como feature adicional
df['risk_score'] = (
    df['pack_years'].fillna(0) +
    df['radon_exposure'] +
    df['asbestos_exposure'] +
    df['secondhand_smoke_exposure'] +
    df['copd_diagnosis'] +
    df['alcohol_consumption'] +
    df['family_history']
)

logging.info("Variables categóricas codificadas y risk_score calculado")

features = ['age','pack_years','risk_score','gender','copd_diagnosis',
            'alcohol_consumption','family_history','asbestos_exposure',
            'secondhand_smoke_exposure','radon_exposure']

X = df[features]
y = df['lung_cancer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

logging.info(f"Train: {X_train.shape[0]} muestras, Test: {X_test.shape[0]} muestras")
logging.info(f"Distribución original train: {y_train.value_counts().to_dict()}")

# Imputar NaN en X_train antes de SMOTE
X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_train.mean(numeric_only=True))

# SMOTE para balance de clases
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

logging.info(f"Tamaño tras SMOTE: {X_train_sm.shape}")
logging.info(f"Distribución tras SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

# Entrenar XGBoost con hiperparámetros optimizados
xgb_model_smote = xgb.XGBClassifier(
    colsample_bytree=0.9,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=200,
    reg_alpha=0.1,
    reg_lambda=1,
    subsample=0.8,
    random_state=20,
    n_jobs=-1,
    eval_metric=["logloss", "auc", "aucpr"],
    use_label_encoder=False
)

logging.info("Iniciando entrenamiento del modelo XGBoost con SMOTE...")

xgb_model_smote.fit(
    X_train_sm, y_train_sm,
    eval_set=[(X_test, y_test)],
    verbose=True
)

logging.info("Entrenamiento completado")

# Evaluar el modelo SMOTE
y_proba_sm = xgb_model_smote.predict_proba(X_test)[:,1]
y_pred_sm = (y_proba_sm >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred_sm)
roc_auc = roc_auc_score(y_test, y_proba_sm)
report = classification_report(y_test, y_pred_sm)

print("\n" + "="*60)
print("XGBoost con SMOTE Performance")
print("="*60)
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\n" + report)
print("="*60)

logging.info(f"Accuracy: {acc:.4f}")
logging.info(f"ROC AUC: {roc_auc:.4f}")
logging.info(f"\n{report}")

# Feature importance SMOTE (mejorado visualmente)
plt.figure(figsize=(10,7), facecolor='white')
xgb.plot_importance(xgb_model_smote, importance_type='weight', max_num_features=10, 
                     height=0.6, color='#3498db', title='', grid=False)
plt.title("Importancia de Features (XGBoost SMOTE)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel('F Score (weight)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig(unique_fig_name("feature_importance_smote"), dpi=300, bbox_inches='tight')
plt.close()

# SHAP SMOTE
logging.info("Calculando valores SHAP...")
explainer_sm = shap.TreeExplainer(xgb_model_smote)
shap_values_sm = explainer_sm.shap_values(X_test)

# Resumen global SHAP (mejorado visualmente)
plt.figure(figsize=(10, 7), facecolor='white')
shap.summary_plot(shap_values_sm, X_test, plot_type="bar", show=False, color='#e74c3c')
plt.title("Importancia SHAP - Resumen Global (SMOTE)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(unique_fig_name("shap_summary_bar_smote"), dpi=300, bbox_inches='tight')
plt.close()

# Resumen detallado SHAP (mejorado visualmente)
plt.figure(figsize=(10, 7), facecolor='white')
shap.summary_plot(shap_values_sm, X_test, show=False)
plt.title("Valores SHAP - Impacto de Features (SMOTE)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(unique_fig_name("shap_summary_beeswarm_smote"), dpi=300, bbox_inches='tight')
plt.close()

# Matriz de confusión SMOTE (mejorado visualmente)
cm_sm = confusion_matrix(y_test, y_pred_sm)
plt.figure(figsize=(8,6), facecolor='white')
sns.heatmap(cm_sm, annot=True, fmt='d', cmap='RdYlGn_r', cbar_kws={'label': 'Count'},
            annot_kws={'size': 14, 'weight': 'bold'}, linewidths=2, linecolor='white')
plt.xlabel('Predicción', fontsize=12, fontweight='bold')
plt.ylabel('Real', fontsize=12, fontweight='bold')
plt.title('Matriz de Confusión (SMOTE)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(unique_fig_name("confusion_matrix_smote"), dpi=300, bbox_inches='tight')
plt.close()

logging.info(f"Todas las gráficas SMOTE se guardaron en la carpeta: {output_folder}")

# Guardar el modelo SMOTE
model_path = os.path.join(output_folder, "xgb_model_smote.json")
xgb_model_smote.save_model(model_path)
logging.info(f"El modelo SMOTE se guardó en: {model_path}")

# Permutation Importance (mejorado visualmente)
logging.info("Calculando permutation importance...")
perm_result = permutation_importance(xgb_model_smote, X_test, y_test, n_repeats=30, 
                                     random_state=42, scoring='roc_auc')
importances = perm_result.importances_mean
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 7), facecolor='white')
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(indices)))
plt.barh([features[i] for i in indices][::-1], importances[indices][::-1], color=colors[::-1])
plt.xlabel('Permutation Importance (ROC AUC)', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Importancia por Permutación (SMOTE)', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(unique_fig_name("permutation_importance_smote"), dpi=300, bbox_inches='tight')
plt.close()

logging.info("Permutation importance SMOTE calculada y guardada")
logging.info("Proceso completado exitosamente")

print(f"\n✓ Proceso SMOTE completado. Resultados guardados en: {output_folder}")