import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Copia para no modificar el original
    df = df.copy()

    # Rellenos y mapeos como en el entrenamiento
    if 'alcohol_consumption' in df.columns and df['alcohol_consumption'].isnull().sum() > 0:
        df['alcohol_consumption'] = df['alcohol_consumption'].fillna('Unknown')

    mappings = {
        'gender': {'Male': 0, 'Female': 1},
        'family_history': {'No': 0, 'Yes': 1},
        'copd_diagnosis': {'No': 0, 'Yes': 1},
        'asbestos_exposure': {'No': 0, 'Yes': 1},
        'secondhand_smoke_exposure': {'No': 0, 'Yes': 1},
        'lung_cancer': {'No': 0, 'Yes': 1},
        'alcohol_consumption': {'None': 0, 'Moderate': 1, 'Heavy': 2},
        'radon_exposure': {'Low': 0, 'Medium': 1, 'High': 2},
    }

    for col, mapping in mappings.items():
        if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].map(mapping)

    # risk_score como en entrenamiento
    for need in [
        'pack_years', 'radon_exposure', 'asbestos_exposure',
        'secondhand_smoke_exposure', 'copd_diagnosis', 'alcohol_consumption', 'family_history'
    ]:
        if need not in df.columns:
            raise ValueError(f"Falta la columna requerida para el preprocesado: {need}")

    df['risk_score'] = (
        df['pack_years'].fillna(0)
        + df['radon_exposure']
        + df['asbestos_exposure']
        + df['secondhand_smoke_exposure']
        + df['copd_diagnosis']
        + df['alcohol_consumption'].fillna(0)
        + df['family_history']
    )

    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluar modelo XGBoost entrenado o predecir nuevos casos.')
    parser.add_argument('--data', type=str, default=os.path.join(os.path.dirname(__file__), 'lung_cancer_dataset.csv'),
                        help='Ruta al CSV con datos. Por defecto, lung_cancer_dataset.csv en esta carpeta.')
    parser.add_argument('--model', type=str, default=os.path.join(os.path.dirname(__file__), 'resultados_modelo', 'xgb_model.json'),
                        help='Ruta al modelo XGBoost guardado (.json).')
    parser.add_argument('--threshold', type=float, default=0.5, help='Umbral para clasificar (0-1).')
    parser.add_argument('--outdir', type=str, default=os.path.join(os.path.dirname(__file__), 'resultados_modelo'),
                        help='Carpeta para guardar resultados.')

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Cargar datos y preprocesar
    df = pd.read_csv(args.data)
    df_prep = preprocess(df)

    features = [
        'age', 'pack_years', 'risk_score', 'gender', 'copd_diagnosis',
        'alcohol_consumption', 'family_history', 'asbestos_exposure',
        'secondhand_smoke_exposure', 'radon_exposure'
    ]
    X = df_prep[features]

    # Cargar modelo
    model = xgb.XGBClassifier()
    model.load_model(args.model)

    # Predicciones
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= args.threshold).astype(int)

    # Si existe la etiqueta real, evaluamos
    metrics_txt_path = os.path.join(args.outdir, 'eval_resumen.txt')
    if 'lung_cancer' in df_prep.columns:
        y_true = df_prep['lung_cancer'].astype(int)
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = float('nan')
        report = classification_report(y_true, y_pred)

        with open(metrics_txt_path, 'w', encoding='utf-8') as f:
            f.write('=== Evaluación del modelo ===\n')
            f.write(f'Accuracy: {acc:.4f}\n')
            f.write(f'ROC AUC: {auc:.4f}\n')
            f.write('\n')
            f.write(report)

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (evaluación)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'confusion_matrix_eval.png'))
        plt.close()

        print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")
        print(f"Reporte de clasificación y matriz de confusión guardados en: {args.outdir}")
    else:
        print('Columna objetivo `lung_cancer` no encontrada. Se generarán solo predicciones.')

    # Guardar predicciones
    out_pred = df.copy()
    out_pred['pred_proba'] = y_proba
    out_pred['pred_label'] = y_pred
    out_csv = os.path.join(args.outdir, 'predicciones.csv')
    out_pred.to_csv(out_csv, index=False)
    print(f"Predicciones guardadas en: {out_csv}")


if __name__ == '__main__':
    main()
