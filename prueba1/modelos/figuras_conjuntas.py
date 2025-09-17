import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Definir la red igual que en el entrenamiento
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Cargar modelo entrenado
model = Net()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Preparar datos de test
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Obtener un batch de imágenes y predicciones
images, labels = next(iter(test_loader))
outputs = model(images)
preds = outputs.argmax(dim=1)

# Mostrar las primeras 16 imágenes con su etiqueta real y predicha
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    img = images[i].squeeze().numpy() * 0.3081 + 0.1307  # Desnormaliza
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Real: {labels[i].item()}\nPred: {preds[i].item()}')
    ax.axis('off')
plt.tight_layout()
plt.show()
