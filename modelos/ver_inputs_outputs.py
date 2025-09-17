import torch
import matplotlib.pyplot as plt

# Cargar los archivos guardados
test_inputs = torch.load('test_inputs.pt')
test_outputs = torch.load('test_outputs.pt')

print('Forma de los inputs:', test_inputs.shape)
print('Forma de los outputs:', test_outputs.shape)

# Mostrar el primer input como imagen (desnormalizando)
img = test_inputs[0].squeeze().numpy() * 0.3081 + 0.1307
plt.imshow(img, cmap='gray')
plt.title('Primer input (imagen)')
plt.show()

# Mostrar el primer output (vector de logits)
print('Primer output (logits):', test_outputs[0])
