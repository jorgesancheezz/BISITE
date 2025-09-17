import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Modelos
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

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Función para evaluar accuracy de un modelo
def eval_model(model_class, weights_path, seeds=[0,1,2,3,4,5,6,7,8,9]):
    accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_class()
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset)
        accs.append(acc)
    return np.array(accs)

# Evalúa ambos modelos (ajusta los nombres de los archivos de pesos si es necesario)
acc_fc = eval_model(FCNet, 'model.pth')
acc_cnn = eval_model(Mnist_CNN, 'model_conv.pth')

diffs = acc_cnn - acc_fc
mean_diff = np.mean(diffs)
J = len(diffs)
s2 = np.var(diffs, ddof=1)
n_test = 10000  # MNIST test
n_train = 60000  # MNIST train
var_corr = (1/J + n_test/n_train) * s2
t_stat = mean_diff / np.sqrt(var_corr)
p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=J-1))

print(f'T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}')

# Visualización
plt.figure(figsize=(8,4))
plt.plot(acc_fc, 'o-', label='Fully Connected')
plt.plot(acc_cnn, 's-', label='Convolucional')
plt.ylim(0.97, 0.975)
plt.title('Accuracy de los modelos por ejecución')
plt.xlabel('Ejecución')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

txt = f"T={t_stat:.2f}, p-value={p_value:.4f}\n"
if p_value < 0.05:
    txt += "¡Diferencia significativa!"
else:
    txt += "No hay diferencia significativa."
print(txt)
