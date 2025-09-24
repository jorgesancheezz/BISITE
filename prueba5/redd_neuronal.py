import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
# Crear carpeta para guardar resultados dentro de prueba5
output_folder = os.path.join(os.path.dirname(__file__), "resultados_red_neuronal")
os.makedirs(output_folder, exist_ok=True)

# Utilidad para nombre único de imagen
def unique_fig_name(base):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_folder, f"{base}_{timestamp}.png")

# 1. Cargar y preparar los datos 
data_path = "C:\\Users\\BISITE-NEL\\Desktop\\pruebas\\prueba5\\lung_cancer_dataset.csv"
df = pd.read_csv(data_path, keep_default_na=False) # Para que no transforme None en NaN

# Codificar variables categóricas igual que en tu flujo principal
df['gender'] = df['gender'].map({'Male':0,'Female':1})
df['family_history'] = df['family_history'].map({'No':0,'Yes':1})
df['copd_diagnosis'] = df['copd_diagnosis'].map({'No':0,'Yes':1})
df['asbestos_exposure'] = df['asbestos_exposure'].map({'No':0,'Yes':1})
df['secondhand_smoke_exposure'] = df['secondhand_smoke_exposure'].map({'No':0,'Yes':1})
df['lung_cancer'] = df['lung_cancer'].map({'No':0,'Yes':1})
df['alcohol_consumption'] = df['alcohol_consumption'].map({'None': 0, 'Moderate': 1, 'Heavy': 2})
df['radon_exposure'] = df['radon_exposure'].map({'Low':0,'Medium':1,'High':2})

features = ['age','pack_years','gender','copd_diagnosis',
            'alcohol_consumption','family_history','asbestos_exposure',
            'secondhand_smoke_exposure','radon_exposure']

X = df[features]
y = df['lung_cancer']

# 2. Dividir en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Preprocesar los datos: escalar y convertir a tensores
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 4. Definir la arquitectura de la red 
class RedNet(nn.Module):
    def __init__(self, input_dim):
        super(RedNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# 5. Instanciar la red, función de pérdida y optimizador
input_dim = X_train.shape[1]
model = RedNet(input_dim)
criterion = nn.BCEWithLogitsLoss()  # Mayor estabilidad
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Entrenamiento de la red neuronal
epochs = 6000
for epoch in range(epochs):
    model.train()  # Modo entrenamiento
    optimizer.zero_grad()  # Reiniciar gradientes
    outputs = model(X_train_tensor)  
    loss = criterion(outputs, y_train_tensor)  # Calcular pérdida
    loss.backward()  
    optimizer.step()  # Actualizar pesos
    if (epoch+1) % 500 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# 7. Evaluación en el conjunto de test
model.eval()  # Modo evaluación
with torch.no_grad():
    logits = model(X_test_tensor)  # Salida sin sigmoide
    y_pred_prob = torch.sigmoid(logits)  # Probabilidades entre 0 y 1
    y_pred = (y_pred_prob >= 0.5).float()  # Umbral 0.5 para clase positiva
    accuracy = (y_pred == y_test_tensor).float().mean().item()
    print(f'Accuracy en test (PyTorch): {accuracy:.4f}')

# Guardar modelo para inferencia
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = os.path.join(output_folder, f"rednet_{ts}.pth")
print(f"[OK] Modelo guardado: {model_path}")
