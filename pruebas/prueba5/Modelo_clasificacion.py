import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import logging

# Crear carpeta para guardar resultados dentro de prueba5
output_folder = os.path.join(os.path.dirname(__file__), "resultados_modelo")
os.makedirs(output_folder, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_folder, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logging.info("Inicio del entrenamiento del modelo")

# Utilidad para nombre único de imagen
def unique_fig_name(base):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_folder, f"{base}_{timestamp}.png")

# Cargar y preparar los datos
data_path = os.path.join(os.path.dirname(__file__), "lung_cancer_dataset.csv")
df = pd.read_csv(data_path)

logging.info(f"Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")

if df['alcohol_consumption'].isnull().sum() > 0:
    df['alcohol_consumption'] = df['alcohol_consumption'].fillna('Unknown')
    
# Codificar variables categóricas
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

# Definir features y target
features = ['age','pack_years','risk_score','gender','copd_diagnosis',
            'alcohol_consumption','family_history','asbestos_exposure',
            'secondhand_smoke_exposure','radon_exposure']

X = df[features]
y = df['lung_cancer']

# Dividir en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

logging.info(f"Train: {X_train.shape[0]} muestras, Test: {X_test.shape[0]} muestras")
logging.info(f"Distribución train: {y_train.value_counts().to_dict()}")
logging.info(f"Distribución test: {y_test.value_counts().to_dict()}")

# Calcular pesos considerando tanto el balance de clases como el risk_score
# Los casos de alto riesgo tendrán más peso en el entrenamiento
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
logging.info(f'Pesos de clases base: {class_weight_dict}')

# Aplicar pesos de clase
train_sample_weight = y_train.map(class_weight_dict).values

# Ajustar los pesos según el risk_score (mayor riesgo = mayor peso)
# Normalizar risk_score para que esté entre 0.8 y 1.2 (rango conservador para mantener pesos positivos)
risk_scores_train = X_train['risk_score'].values
risk_min = risk_scores_train.min()
risk_max = risk_scores_train.max()
# Normalizar a [0, 1] y luego escalar a [0.8, 1.2]
if risk_max > risk_min:
    risk_weight_factor = 0.8 + 0.4 * ((risk_scores_train - risk_min) / (risk_max - risk_min))
else:
    risk_weight_factor = np.ones_like(risk_scores_train)
train_sample_weight = train_sample_weight * risk_weight_factor

val_sample_weight = y_test.map(class_weight_dict).values
risk_scores_test = X_test['risk_score'].values
# Usar los mismos valores min/max del entrenamiento para consistencia
if risk_max > risk_min:
    risk_weight_factor_test = 0.8 + 0.4 * ((risk_scores_test - risk_min) / (risk_max - risk_min))
else:
    risk_weight_factor_test = np.ones_like(risk_scores_test)
val_sample_weight = val_sample_weight * risk_weight_factor_test

logging.info("Pesos de muestra ajustados por risk_score")

# Entrenar y definir XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=202,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=20,
    reg_lambda=1,
    reg_alpha=0,
    n_jobs=-1,
    eval_metric=["logloss", "auc", "aucpr"],
    use_label_encoder=False
)

logging.info("Iniciando entrenamiento del modelo XGBoost...")

xgb_model.fit(
    X_train, y_train,
    sample_weight=train_sample_weight,
    eval_set=[(X_test, y_test)],
    sample_weight_eval_set=[val_sample_weight],
    verbose=True
)

logging.info("Entrenamiento completado")

# Evaluar el modelo
y_proba = xgb_model.predict_proba(X_test)[:,1]
y_pred = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)

print("\n" + "="*60)
print("XGBoost Performance")
print("="*60)
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\n" + report)
print("="*60)

logging.info(f"Accuracy: {acc:.4f}")
logging.info(f"ROC AUC: {roc_auc:.4f}")
logging.info(f"\n{report}")

# Feature importance nativa de XGBoost (mejorado visualmente)
plt.figure(figsize=(10,7), facecolor='white')
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10, 
                     height=0.6, color='#2ecc71', title='', grid=False)
plt.title("Importancia de Features (XGBoost)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel('F Score (weight)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig(unique_fig_name("feature_importance"), dpi=300, bbox_inches='tight')
plt.close()

# Explicabilidad con SHAP
logging.info("Calculando valores SHAP...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Resumen global (barras) - mejorado visualmente
plt.figure(figsize=(10, 7), facecolor='white')
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, color='#3498db')
plt.title("Importancia SHAP - Resumen Global", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(unique_fig_name("shap_summary_bar"), dpi=300, bbox_inches='tight')
plt.close()

# Resumen detallado (beeswarm) - mejorado visualmente
plt.figure(figsize=(10, 7), facecolor='white')
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Valores SHAP - Impacto de Features", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(unique_fig_name("shap_summary_beeswarm"), dpi=300, bbox_inches='tight')
plt.close()

# Matriz de confusión - mejorado visualmente
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6), facecolor='white')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
            annot_kws={'size': 14, 'weight': 'bold'}, linewidths=2, linecolor='white')
plt.xlabel('Predicción', fontsize=12, fontweight='bold')
plt.ylabel('Real', fontsize=12, fontweight='bold')
plt.title('Matriz de Confusión', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(unique_fig_name("confusion_matrix"), dpi=300, bbox_inches='tight')
plt.close()

logging.info(f"Todas las gráficas se guardaron en la carpeta: {output_folder}")

# Guardar el modelo entrenado
model_path = os.path.join(output_folder, "xgb_model.json")
xgb_model.save_model(model_path)
logging.info(f"El modelo entrenado se guardó en: {model_path}")

# Permutation Importance manual
def permutation_importance_manual(xgb_model, X, y, accuracy_score, n_repeats=20, random_state=42):
    np.random.seed(random_state)
    base_score = accuracy_score(y, xgb_model.predict(X))
    importances = []
    for col in X.columns:
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)
            score = accuracy_score(y, xgb_model.predict(X_permuted))
            scores.append(base_score - score)
        importances.append(np.mean(scores))
    return np.array(importances)

logging.info("Calculando permutation importance manual...")

# Graficar y guardar permutation importance manual - mejorado visualmente
importances = permutation_importance_manual(xgb_model, X_test, y_test, accuracy_score, n_repeats=20)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 7), facecolor='white')
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
plt.barh([X_test.columns[i] for i in indices][::-1], importances[indices][::-1], color=colors[::-1])
plt.xlabel('Permutation Importance', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Importancia por Permutación', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(unique_fig_name("permutation_importance_manual"), dpi=300, bbox_inches='tight')
plt.close()

logging.info("Permutation importance manual calculada y guardada")
logging.info("Proceso completado exitosamente")

print(f"\n✓ Proceso completado. Resultados guardados en: {output_folder}")