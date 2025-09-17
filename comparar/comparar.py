import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

# Modelo fully connected (mnist_pytorch)
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Modelo convolucional (modelo_conv)
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

# Cargar modelos entrenados
fc_model = FCNet()
fc_model.load_state_dict(torch.load('model.pth'))
fc_model.eval()

cnn_model = Mnist_CNN()
cnn_model.load_state_dict(torch.load('model_conv.pth'))
cnn_model.eval()

# Preparar datos de test
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Obtener un batch de imágenes y etiquetas
images, labels = next(iter(test_loader))

# Predicciones de ambos modelos
with torch.no_grad():
    fc_outputs = fc_model(images)
    fc_preds = fc_outputs.argmax(dim=1)
    cnn_outputs = cnn_model(images)
    cnn_preds = cnn_outputs.argmax(dim=1)

# Mostrar resultados comparados
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i in range(16):
    img = images[i].squeeze().numpy() * 0.3081 + 0.1307
    axes[i//4, 2*(i%4)].imshow(img, cmap='gray')
    axes[i//4, 2*(i%4)].set_title(f'Real: {labels[i].item()}')
    axes[i//4, 2*(i%4)].axis('off')
    axes[i//4, 2*(i%4)+1].text(0.5, 0.7, f'FC: {fc_preds[i].item()}', fontsize=12, ha='center')
    axes[i//4, 2*(i%4)+1].text(0.5, 0.3, f'CNN: {cnn_preds[i].item()}', fontsize=12, ha='center')
    axes[i//4, 2*(i%4)+1].set_xticks([])
    axes[i//4, 2*(i%4)+1].set_yticks([])
    axes[i//4, 2*(i%4)+1].set_frame_on(False)
plt.tight_layout()
plt.show()
