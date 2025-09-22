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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import logging
from datetime import datetime

# Crear carpeta para guardar resultados dentro de prueba5
output_folder = os.path.join(os.path.dirname(__file__), "resultados_modelo_smote")
os.makedirs(output_folder, exist_ok=True)

# Utilidad para nombre único de imagen
def unique_fig_name(base):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_folder, f"{base}_{timestamp}.png")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_folder, "smote_log.log")),
        logging.StreamHandler()
    ]
)
logging.info("Inicio del script SMOTE")

# Cargar y preparar los datos

data_path = "C:\\Users\\BISITE-NEL\\Desktop\\pruebas\\prueba5\\lung_cancer_dataset.csv"
df = pd.read_csv(data_path)

if df['alcohol_consumption'].isnull().sum() > 0:
    df['alcohol_consumption'] = df['alcohol_consumption'].fillna('Unknown')
df['gender'] = df['gender'].map({'Male':0,'Female':1})
df['family_history'] = df['family_history'].map({'No':0,'Yes':1})
df['copd_diagnosis'] = df['copd_diagnosis'].map({'No':0,'Yes':1})
df['asbestos_exposure'] = df['asbestos_exposure'].map({'No':0,'Yes':1})
df['secondhand_smoke_exposure'] = df['secondhand_smoke_exposure'].map({'No':0,'Yes':1})
df['lung_cancer'] = df['lung_cancer'].map({'No':0,'Yes':1})
df['alcohol_consumption'] = df['alcohol_consumption'].map({'None':0,'Moderate':1,'Heavy':1})
df['radon_exposure'] = df['radon_exposure'].map({'Low':0,'Medium':1,'High':2})

features = ['age','pack_years','gender','copd_diagnosis',
            'alcohol_consumption','family_history','asbestos_exposure',
            'secondhand_smoke_exposure','radon_exposure']

X = df[features]
y = df['lung_cancer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Imputar NaN en X_train antes de SMOTE
X_train = X_train.fillna(X_train.mean(numeric_only=True))
# ----------- SMOTE -------------
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

logging.info(f"Tamaño tras SMOTE: {X_train_sm.shape}")



# ----------- Entrenar XGBoost con los mejores hiperparámetros encontrados -----------
xgb_model_smote = xgb.XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    reg_alpha=0,
    reg_lambda=1,
    subsample=0.8,
    random_state=20,
    n_jobs=-1,
    eval_metric=["logloss", "auc", "aucpr"],
    use_label_encoder=False
)
xgb_model_smote.fit(
    X_train_sm, y_train_sm,
    eval_set=[(X_test, y_test)],
    verbose=True
)
logging.info("Modelo entrenado con hiperparámetros óptimos del log. GridSearchCV omitido para acelerar.")

# Evaluar el modelo SMOTE
y_proba_sm = xgb_model_smote.predict_proba(X_test)[:,1]
y_pred_sm = (y_proba_sm >= 0.5).astype(int)

print("\nXGBoost con SMOTE Performance")
print("Accuracy:", accuracy_score(y_test, y_pred_sm))
print("ROC AUC:", roc_auc_score(y_test, y_proba_sm))
print(classification_report(y_test, y_pred_sm))

logging.info(f"Accuracy: {accuracy_score(y_test, y_pred_sm):.4f}")
logging.info(f"ROC AUC: {roc_auc_score(y_test, y_proba_sm):.4f}")
logging.info("\n" + classification_report(y_test, y_pred_sm))

# Feature importance SMOTE
plt.figure(figsize=(8,6))
xgb.plot_importance(xgb_model_smote, importance_type='weight', max_num_features=10, height=0.5, color='blue')
plt.title("Feature Importance (XGBoost SMOTE)")
plt.tight_layout()
plt.savefig(unique_fig_name("feature_importance_smote"))
plt.close()

# SHAP SMOTE
explainer_sm = shap.TreeExplainer(xgb_model_smote)
shap_values_sm = explainer_sm.shap_values(X_test)
shap.summary_plot(shap_values_sm, X_test, plot_type="bar", show=False)
plt.savefig(unique_fig_name("shap_summary_bar_smote"))
plt.close()
shap.summary_plot(shap_values_sm, X_test, show=False)
plt.savefig(unique_fig_name("shap_summary_beeswarm_smote"))
plt.close()

# Matriz de confusión SMOTE
cm_sm = confusion_matrix(y_test, y_pred_sm)
plt.figure(figsize=(5,4))
sns.heatmap(cm_sm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (SMOTE)')
plt.savefig(unique_fig_name("confusion_matrix_smote"))

print(f"Todas las gráficas SMOTE se guardaron en la carpeta: {output_folder}")
# Guardar el modelo SMOTE
xgb_model_smote.save_model(os.path.join(output_folder, "xgb_model_smote.json"))

print(f"El modelo SMOTE se guardó en: {os.path.join(output_folder, 'xgb_model_smote.json')}")

# Permutation Importance (solo al final)
from sklearn.inspection import permutation_importance
perm_result = permutation_importance(xgb_model_smote, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc')
importances = perm_result.importances_mean
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8,6))
plt.barh([features[i] for i in indices], importances[indices])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance by Permutation (SMOTE)')
plt.tight_layout()
plt.savefig(unique_fig_name("permutation_importance_smote"))
plt.close()
print("Permutation importance SMOTE calculada y guardada.")
