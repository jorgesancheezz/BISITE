# --- Importar librerías ---
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Cargar y preparar los datos
# -----------------------------
data_path = "C:\\Users\\BISITE-NEL\\Desktop\\pruebas\\prueba5\\lung_cancer_dataset.csv"
df = pd.read_csv(data_path)

# Codificar variables categóricas
df['gender'] = df['gender'].map({'Male':0,'Female':1})
df['family_history'] = df['family_history'].map({'No':0,'Yes':1})
df['copd_diagnosis'] = df['copd_diagnosis'].map({'No':0,'Yes':1})
df['asbestos_exposure'] = df['asbestos_exposure'].map({'No':0,'Yes':1})
df['secondhand_smoke_exposure'] = df['secondhand_smoke_exposure'].map({'No':0,'Yes':1})
df['lung_cancer'] = df['lung_cancer'].map({'No':0,'Yes':1})
df['alcohol_consumption'] = df['alcohol_consumption'].map({'None':0,'Moderate':1,'Heavy':2})
df['radon_exposure'] = df['radon_exposure'].map({'Low':0,'Medium':1,'High':2})

# Crear un 'risk_score' combinando factores de riesgo numéricos y categóricos
df['risk_score'] = df['pack_years'].fillna(0) + df['radon_exposure'] + df['asbestos_exposure'] + df['secondhand_smoke_exposure'] + df['copd_diagnosis'] + df['alcohol_consumption'].fillna(0) + df['family_history']

# -----------------------------
# 2️⃣ Definir features y target
# -----------------------------
features = ['age','pack_years','risk_score','gender','copd_diagnosis',
            'alcohol_consumption','family_history','asbestos_exposure',
            'secondhand_smoke_exposure','radon_exposure']

X = df[features]
y = df['lung_cancer']

# -----------------------------
# 3️⃣ Dividir en entrenamiento y test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4️⃣ Entrenar modelo XGBoost
# -----------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# -----------------------------
# 5️⃣ Evaluar el modelo
# -----------------------------
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:,1]

print("=== XGBoost Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# -----------------------------
# 6️⃣ Feature importance nativa de XGBoost
# -----------------------------
plt.figure(figsize=(8,6))
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10, height=0.5, color='green')
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()

# -----------------------------
# 7️⃣ Explicabilidad con SHAP
# -----------------------------
# Crear explainer para árbol
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Resumen global: qué features son más importantes en promedio
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Resumen detallado: cómo cada feature afecta cada predicción
shap.summary_plot(shap_values, X_test)

# -----------------------------
# 8️⃣ Ejemplo de explicación individual
# -----------------------------
# Elegimos el primer paciente del test set
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
