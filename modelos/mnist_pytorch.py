import torch  # Importa la librería principal de PyTorch
import torch.nn as nn  # Importa el módulo de redes neuronales
import torch.optim as optim  # Importa los optimizadores
from torchvision import datasets, transforms  # Importa utilidades para datasets y transformaciones
from torch.utils.data import DataLoader  # Importa el DataLoader para manejar los datos por lotes

# Define las transformaciones: convierte a tensor y normaliza los datos de MNIST
transform = transforms.Compose([
    transforms.ToTensor(),  # Convierte la imagen a tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normaliza con media y desviación estándar de MNIST
])

# Descarga y carga el dataset MNIST para entrenamiento y prueba
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # Datos de entrenamiento
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # Datos de prueba

# Crea los DataLoaders para manejar los datos en lotes
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Lotes de 64 para entrenamiento
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)  # Lotes de 1000 para prueba

# Define una red neuronal simple
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # Inicializa la clase base
        self.fc1 = nn.Linear(28*28, 128)  # Capa totalmente conectada: de 784 a 128
        self.fc2 = nn.Linear(128, 64)     # Capa totalmente conectada: de 128 a 64
        self.fc3 = nn.Linear(64, 10)      # Capa totalmente conectada: de 64 a 10 (clases)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Aplana la imagen de 28x28 a un vector de 784
        x = torch.relu(self.fc1(x))  # Aplica ReLU a la primera capa
        x = torch.relu(self.fc2(x))  # Aplica ReLU a la segunda capa
        x = self.fc3(x)  # Capa de salida (sin activación, ya que CrossEntropyLoss la incluye)
        return x  # Devuelve la salida

model = Net()  # Instancia el modelo

# Define la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()  # Pérdida para clasificación multiclase
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizador Adam con tasa de aprendizaje 0.001

# Función para entrenar el modelo
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # Pone el modelo en modo entrenamiento
    for batch_idx, (data, target) in enumerate(train_loader):  # Itera sobre los lotes
        data, target = data.to(device), target.to(device)  # Mueve datos y etiquetas al dispositivo
        optimizer.zero_grad()  # Reinicia los gradientes
        output = model(data)  # Calcula la salida del modelo
        loss = criterion(output, target)  # Calcula la pérdida
        loss.backward()  # Propaga el error hacia atrás
        optimizer.step()  # Actualiza los pesos
        if batch_idx % 100 == 0:  # Cada 100 lotes imprime el progreso
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}')

# Función para evaluar el modelo
def test(model, device, test_loader, criterion):
    model.eval()  # Pone el modelo en modo evaluación
    test_loss = 0  # Acumula la pérdida
    correct = 0  # Cuenta los aciertos
    with torch.no_grad():  # No calcula gradientes (más eficiente)
        for data, target in test_loader:  # Itera sobre los lotes de prueba
            data, target = data.to(device), target.to(device)  # Mueve datos y etiquetas al dispositivo
            output = model(data)  # Calcula la salida del modelo
            test_loss += criterion(output, target).item()  # Suma la pérdida
            pred = output.argmax(dim=1, keepdim=True)  # Obtiene la clase predicha
            correct += pred.eq(target.view_as(pred)).sum().item()  # Suma los aciertos
    test_loss /= len(test_loader.dataset)  # Promedia la pérdida
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')  # Imprime resultados

# Configura el dispositivo: usa GPU si está disponible, si no CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Mueve el modelo al dispositivo

# Entrena y evalúa el modelo por 5 épocas
for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, criterion, epoch) 
    test(model, device, test_loader, criterion)  

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'model.pth')  # Guarda solo los pesos del modelo

# Guardar los primeros 100 inputs y outputs del set de test
# Se guardan solo 100 para evitar archivos muy grandes y porque suele ser suficiente para análisis o depuración.
inputs_list = []
outputs_list = []
count = 0
max_save = 100
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        if count < max_save:
            n_to_save = min(max_save - count, data.size(0))
            inputs_list.append(data[:n_to_save].cpu())
            outputs_list.append(output[:n_to_save].cpu())
            count += n_to_save
        if count >= max_save:
            break
if inputs_list and outputs_list:
    all_inputs = torch.cat(inputs_list, dim=0)
    all_outputs = torch.cat(outputs_list, dim=0)
    torch.save(all_inputs, 'test_inputs.pt')
    torch.save(all_outputs, 'test_outputs.pt')
