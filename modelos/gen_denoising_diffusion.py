import torch
from torchvision import utils
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import matplotlib.pyplot as plt
import argparse
import os


# Argumentos para la generación

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='diffusion_mnist.pth', help='Ruta al modelo entrenado')
parser.add_argument('--out_dir', type=str, default='resultados_diffusion', help='Directorio de salida para las imágenes generadas')
parser.add_argument('--batch_size', type=int, default=16, help='Número de imágenes a generar')
args = parser.parse_args()

# Parámetros


image_size = 28
timesteps = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Modelo Unet no condicional
model = Unet(
    dim=image_size,
    channels=1,
    dim_mults=(1, 2, 4),
    flash_attn=False
).to(device)


# Cargar pesos entrenados
if os.path.exists(args.model_path):
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Modelo cargado de {args.model_path}")
else:
    print(f"Advertencia: No se encontró el modelo {args.model_path}. Se usará el modelo sin entrenar.")

diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=50
).to(device)


# Generar imágenes no condicionales
os.makedirs(args.out_dir, exist_ok=True)
with torch.no_grad():
    samples = diffusion.sample(batch_size=args.batch_size)
    grid = utils.make_grid(samples, nrow=4, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.title('Imágenes generadas (no condicional)')
    plt.imshow(grid.permute(1,2,0).cpu().numpy(), cmap='gray')
    plt.savefig(f'{args.out_dir}/sample.png')
    plt.close()

print(f'Imágenes generadas guardadas en {args.out_dir}/')
