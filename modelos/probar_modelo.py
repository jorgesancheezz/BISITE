import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Definir la clase Net igual que en el archivo de entrenamiento
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instanciar el modelo y cargar los pesos
model = Net()
model.load_state_dict(torch.load('model.pth'))  # Carga los pesos guardados
model.eval()  # Pone el modelo en modo evaluación

print("Modelo cargado y listo para usar.")

# Ejemplo: cargar una imagen del set de prueba MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
img, label = test_dataset[9]  # Toma la novena imagen del set de prueba

# Añade una dimensión para el batch
img = img.unsqueeze(0)

# Pasa la imagen por el modelo
output = model(img)
pred = output.argmax(dim=1, keepdim=True)

print(f'Predicción: {pred.item()} (Etiqueta real: {label})')
