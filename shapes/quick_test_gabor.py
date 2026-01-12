# quick_test_gabor.py

import torch
from better_conv_net import BetterConvNet  # Twoja wersja

# Stwórz model z Gabor
model_gabor = BetterConvNet(init_method="gabor")

# Stwórz model z Kaiming (domyślny)
model_kaiming = BetterConvNet(init_method="kaiming")

# Porównaj
print("Model z Gabor - wizualizuj pierwsze filtry...")
filters_gabor = model_gabor.conv1.weight.data[:16, 0, :, :].numpy()

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters_gabor[i], cmap="gray")
    ax.axis("off")
plt.suptitle("Gabor filters")
plt.savefig("gabor_filters.png")
plt.show()
