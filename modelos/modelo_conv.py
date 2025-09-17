import torch  # Importa la librería principal de PyTorch
import torch.nn as nn  # Importa el módulo de redes neuronales
import torch.optim as optim  # Importa los optimizadores
from torchvision import datasets, transforms  # Importa utilidades para datasets y transformaciones
from torch.utils.data import DataLoader  # Importa el DataLoader para manejar los datos por lotes
import torch.nn.functional as F
from torch import optim  # Importa funciones de activación y otras funciones útiles
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

loss_func = F.cross_entropy
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))
lr = 0.1
epochs = 5
model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Bucle estándar de entrenamiento y evaluación en PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    print(f'Epoch {epoch} Training Loss: {running_loss / len(train_loader):.6f}')

    # Evaluación
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    torch.save(model.state_dict(), 'model_conv.pth')