
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os
import matplotlib.pyplot as plt

# Carpeta para guardar resultados
os.makedirs('resultados_diffusion', exist_ok=True)

# Hiperparámetros
batch_size = 32  
epochs = 5      
image_size = 28
timesteps = 50  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carga MNIST local
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Modelo Unet para 1 canal
model = Unet(
    dim=image_size,
    channels=1,
    dim_mults=(1, 2, 4),
    flash_attn=False
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=timesteps
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Entrenamiento

for epoch in range(1, epochs+1):
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        current_batch_size = imgs.size(0)
        loss = diffusion(imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Tamaño real del batch: {current_batch_size}")
    # Muestra y guarda imágenes generadas al final de cada época
    with torch.no_grad():
        sampled = diffusion.sample(batch_size=16)
        grid = utils.make_grid(sampled, nrow=4, normalize=True, value_range=(-1,1))
        plt.figure(figsize=(6,6))
        plt.axis('off')
        plt.title(f'Epoch {epoch} - Imágenes generadas')
        plt.imshow(grid.permute(1,2,0).cpu().numpy(), cmap='gray')
        plt.savefig(f'resultados_diffusion/diffusion_epoch{epoch}.png')
        plt.close()


# Guarda el modelo entrenado
torch.save(model.state_dict(), 'diffusion_mnist.pth')
print('Entrenamiento finalizado. Revisa la carpeta resultados_diffusion para ver las imágenes generadas y diffusion_mnist.pth para el modelo guardado.')