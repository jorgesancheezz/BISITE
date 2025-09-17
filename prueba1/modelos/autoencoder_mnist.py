import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Definición del Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: reduce la dimensión de 784 a 32
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU()
        )
        # Decoder: reconstruye de 32 a 784
        self.decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()  # Para que la salida esté entre 0 y 1
        )
    def forward(self, x):
        x = x.view(-1, 28*28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Prepara los datos de MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Instancia el autoencoder y define optimizador y función de pérdida
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del autoencoder
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data.view(-1, 28*28))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}")

# Visualización: muestra imágenes originales y reconstruidas
model.eval()
dataiter = iter(train_loader)
images, _ = next(dataiter)
with torch.no_grad():
    reconstructed = model(images)

# Muestra las primeras 8 imágenes originales y sus reconstrucciones
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    # Imagen original
    axes[0, i].imshow(images[i].squeeze().numpy(), cmap='gray')
    axes[0, i].set_title('Original')
    axes[0, i].axis('off')
    # Imagen reconstruida
    rec_img = reconstructed[i].view(28, 28).numpy()
    axes[1, i].imshow(rec_img, cmap='gray')
    axes[1, i].set_title('Reconstruida')
    axes[1, i].axis('off')
plt.tight_layout()
plt.savefig('resultados/autoencoder_resultado4.png')
plt.show()
