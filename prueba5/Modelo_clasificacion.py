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

# Crear carpeta para guardar resultados
output_folder = "resultados_modelo"
os.makedirs(output_folder, exist_ok=True)

# Cargar y preparar los datos
data_path = "C:\\Users\\BISITE-NEL\\Desktop\\pruebas\\prueba5\\lung_cancer_dataset.csv"
df = pd.read_csv(data_path)

if df['alcohol_consumption'].isnull().sum() > 0:
    df['alcohol_consumption'] = df['alcohol_consumption'].fillna('Unknown')
# Codificar variables categóricas
df['gender'] = df['gender'].map({'Male':0,'Female':1})
df['family_history'] = df['family_history'].map({'No':0,'Yes':1})
df['copd_diagnosis'] = df['copd_diagnosis'].map({'No':0,'Yes':1})
df['asbestos_exposure'] = df['asbestos_exposure'].map({'No':0,'Yes':1})
df['secondhand_smoke_exposure'] = df['secondhand_smoke_exposure'].map({'No':0,'Yes':1})
df['lung_cancer'] = df['lung_cancer'].map({'No':0,'Yes':1})
df['alcohol_consumption'] = df['alcohol_consumption'].map({'Moderate':0,'Heavy':1})
df['radon_exposure'] = df['radon_exposure'].map({'Low':0,'Medium':1,'High':2})

df['risk_score'] = (
    df['pack_years'].fillna(0) 
    + df['radon_exposure'] 
    + df['asbestos_exposure'] 
    + df['secondhand_smoke_exposure'] 
    + df['copd_diagnosis'] 
    + df['alcohol_consumption'].fillna(0) 
    + df['family_history']
)


#Definir features y target

features = ['age','pack_years','risk_score','gender','copd_diagnosis',
            'alcohol_consumption','family_history','asbestos_exposure',
            'secondhand_smoke_exposure','radon_exposure']

X = df[features]
y = df['lung_cancer']


#Dividir en entrenamiento y test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Manejar el desbalanceo con pesos de clase
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print('Clases de pesos:', class_weight_dict)
train_sample_weight = y_train.map(class_weight_dict)
val_sample_weight = y_test.map(class_weight_dict)


# Entrenar y definir XGBoost

xgb_model = xgb.XGBClassifier(
    n_estimators=202,        # número máximo de árboles, combinado con early stopping
    max_depth=5,              # suficiente para capturar interacciones sin sobreajustar
    learning_rate=0.05,       # más bajo que 0.1, generaliza mejor
    subsample=0.8,            # usa 80% de los datos en cada árbol → evita sobreajuste
    colsample_bytree=0.8,     # usa 80% de las features en cada árbol
    random_state=20,
    reg_lambda=1,             # regularización L2
    reg_alpha=0,              # regularización L1 (puedes probar con >0 si quieres más sparsity)
    n_jobs=-1,                # usa todos los núcleos de la CPU
    eval_metric=["logloss", "auc", "aucpr"],
    use_label_encoder=False
)

xgb_model.fit(
    X_train, y_train,
    sample_weight=train_sample_weight,  # aplicar pesos de clase en train
    eval_set=[(X_test, y_test)],
    sample_weight_eval_set=[val_sample_weight],  # pesos en validación
    verbose=True
)

# Evaluar el modelo
y_proba = xgb_model.predict_proba(X_test)[:,1]
y_pred = (y_proba >= 0.5).astype(int)

print("XGBoost Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# Feature importance nativa de XGBoost
plt.figure(figsize=(8,6))
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10, height=0.5, color='green')
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "feature_importance.png"))
plt.close()

# Explicabilidad con SHAP
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Resumen global (barras)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig(os.path.join(output_folder, "shap_summary_bar.png"))
plt.close()

# Resumen detallado (beeswarm)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(os.path.join(output_folder, "shap_summary_beeswarm.png"))
plt.close()

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))

print(f"Todas las gráficas se guardaron en la carpeta: {output_folder}")
# Guardar el modelo entrenado
xgb_model.save_model(os.path.join(output_folder, "xgb_model.json"))
print(f"El modelo entrenado se guardó en: {os.path.join(output_folder, 'xgb_model.json')}")