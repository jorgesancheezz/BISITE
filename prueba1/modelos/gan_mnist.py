import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Carpeta para guardar resultados
os.makedirs('resultados_gan', exist_ok=True)


# Hiperparámetros
noise_dim = 100
batch_size = 128
epochs = 50
lr = 0.0002  # Mismo learning rate para G y D
label_smooth = 0.9  # Suavizado de etiquetas reales
label_noise = 0.05  # Ruido en etiquetas


# --- Generador equilibrado con BatchNorm ---
class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)


# --- Discriminador equilibrado con BatchNorm ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# Prepara los datos de MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normaliza a [-1, 1]
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator(noise_dim).to(device)
D = Discriminator().to(device)

# Inicialización de pesos
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
G.apply(weights_init)
D.apply(weights_init)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Entrenamiento del GAN

for epoch in range(1, epochs+1):
    for i, (real_imgs, _) in enumerate(train_loader):
        real_imgs = real_imgs.view(-1, 28*28).to(device)
        batch_size = real_imgs.size(0)
        # Etiquetas reales y falsas con suavizado y ruido
        real_labels = torch.full((batch_size, 1), label_smooth, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        # Añade ruido a las etiquetas reales
        real_labels += label_noise * torch.rand_like(real_labels)
        # --- Entrena Discriminador ---
        z = torch.randn(batch_size, noise_dim, device=device)
        fake_imgs = G(z)
        d_real = D(real_imgs)
        d_fake = D(fake_imgs.detach())
        real_loss = criterion(d_real, real_labels)
        fake_loss = criterion(d_fake, fake_labels)
        d_loss = real_loss + fake_loss
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        # --- Entrena Generador (1:1) ---
        z = torch.randn(batch_size, noise_dim, device=device)
        fake_imgs = G(z)
        # Etiquetas reales para el generador (quiere engañar al D)
        g_loss = criterion(D(fake_imgs), real_labels)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
    print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    # Guarda imágenes generadas cada época
    with torch.no_grad():
        z = torch.randn(8, noise_dim, device=device)
        samples = G(z).cpu().view(-1, 28, 28)
        fig, axes = plt.subplots(1, 8, figsize=(16,2))
        for j in range(8):
            axes[j].imshow(samples[j], cmap='gray', vmin=-1, vmax=1)
            axes[j].axis('off')
        plt.suptitle(f'Epoch {epoch} - Imágenes generadas por el GAN')
        plt.tight_layout()
        plt.savefig(f'resultados_gan/gan_epoch{epoch}.png')
        plt.close()

print('Entrenamiento finalizado. Revisa la carpeta resultados_gan para ver las imágenes generadas.')
