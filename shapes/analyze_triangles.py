import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. DATASET CLASS
# ============================================================================


class TriangleDataset(Dataset):
    """
    Dataset do wczytywania trójkątów z folderu.
    """

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: ścieżka do folderu z obrazami
            transform: transformacje do zastosowania na obrazach
        """
        self.data_dir = data_dir
        self.transform = transform

        # Znajdź wszystkie pliki PNG
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]
        self.image_files.sort()  # dla reprodukowalności

        print(f"Znaleziono {len(self.image_files)} obrazów w {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Wczytaj obraz
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Zastosuj transformacje
        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]  # zwróć także nazwę dla debugowania


# ============================================================================
# 2. NAJPROSTSZA SIEĆ
# ============================================================================


class SimpleConvNet(nn.Module):
    """
    Prosta sieć konwolucyjna - 3 warstwy conv + pooling.
    """

    def __init__(self):
        super(SimpleConvNet, self).__init__()

        # Warstwa 1: wykrywanie krawędzi
        self.conv1 = nn.Conv2d(
            in_channels=3,  # RGB
            out_channels=16,  # 16 filtrów
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512 -> 256

        # Warstwa 2: kompozycje krawędzi
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256 -> 128

        # Warstwa 3: złożone wzorce (tu będziemy szukać detektorów kształtów)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64

        self.relu = nn.ReLU()

        # Globalne pooling na końcu (zamiast FC layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Conv2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Conv3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # Global pooling -> wektor 64-wymiarowy
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # flatten do [batch_size, 64]

        return x


# ============================================================================
# 3. EKSTRAKCJA AKTYWACJI
# ============================================================================


class ActivationExtractor:
    """
    Klasa do wyciągania aktywacji z wybranych warstw.
    """

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def register_hooks(self, layer_names):
        """
        Rejestruje hooki na podanych warstwach.

        Args:
            layer_names: lista nazw warstw, np. ['conv1', 'conv2', 'conv3']
        """
        for name in layer_names:
            layer = getattr(self.model, name)
            hook = layer.register_forward_hook(self._get_activation(name))
            self.hooks.append(hook)

    def _get_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()

        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# 4. GŁÓWNA FUNKCJA ANALIZY
# ============================================================================


def analyze_network(data_dir, batch_size=8, num_workers=2):
    """
    Główna funkcja: wczytaj dane, przepuść przez sieć, zbierz aktywacje.
    """

    # Przygotuj transformacje
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Konwertuj do tensora [0, 1]
            # Możesz dodać normalizację jeśli chcesz:
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )

    # Stwórz dataset i dataloader
    dataset = TriangleDataset(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # False żeby zachować kolejność
        num_workers=num_workers,
    )

    # Stwórz model (losowe wagi!)
    model = SimpleConvNet()
    model.eval()  # tryb ewaluacji (wyłącz dropout gdyby był)

    print(f"\nArchitektura sieci:")
    print(model)
    print(f"\nLiczba parametrów: {sum(p.numel() for p in model.parameters()):,}")

    # Przygotuj ekstraktor aktywacji
    extractor = ActivationExtractor(model)
    extractor.register_hooks(["conv1", "conv2", "conv3"])

    # Zbierz aktywacje ze wszystkich obrazów
    all_activations = {"conv1": [], "conv2": [], "conv3": [], "output": []}
    all_filenames = []

    print(f"\nPrzetwarzam obrazy...")
    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(dataloader):
            # Forward pass
            output = model(images)

            # Zapisz aktywacje
            for layer_name in ["conv1", "conv2", "conv3"]:
                act = extractor.activations[layer_name]
                # Uśrednij po wymiarach przestrzennych -> [batch, channels]
                act_pooled = act.mean(dim=[2, 3])
                all_activations[layer_name].append(act_pooled.numpy())

            all_activations["output"].append(output.numpy())
            all_filenames.extend(filenames)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Przetworzono {(batch_idx + 1) * batch_size} obrazów")

    # Połącz batche
    for key in all_activations:
        all_activations[key] = np.vstack(all_activations[key])

    print(f"\nZebrano aktywacje:")
    for key, value in all_activations.items():
        print(f"  {key}: shape {value.shape}")

    # Cleanup
    extractor.remove_hooks()

    return all_activations, all_filenames, model


# ============================================================================
# 5. PROSTA WIZUALIZACJA
# ============================================================================


def visualize_activations(activations, filenames, n_samples=5):
    """
    Wizualizacja aktywacji dla kilku przykładowych obrazów.
    """
    layers = ["conv1", "conv2", "conv3", "output"]
    n_layers = len(layers)

    fig, axes = plt.subplots(n_samples, n_layers, figsize=(15, 3 * n_samples))

    for i in range(n_samples):
        for j, layer in enumerate(layers):
            ax = axes[i, j] if n_samples > 1 else axes[j]

            # Weź aktywację dla i-tego obrazu
            act = activations[layer][i]

            # Wizualizuj jako barplot
            ax.bar(range(len(act)), act)
            ax.set_title(f"{layer}\n{filenames[i][:20]}...", fontsize=8)
            ax.set_xlabel("Neuron index")
            ax.set_ylabel("Activation")

    plt.tight_layout()
    plt.savefig("activations_visualization_triangles.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Zapisano: activations_visualization_triangles.png")


def visualize_neuron_statistics(activations):
    """
    Statystyki aktywacji neuronów w różnych warstwach.
    """
    layers = ["conv1", "conv2", "conv3", "output"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        act = activations[layer]  # shape: [n_images, n_neurons]

        # Średnia aktywacja każdego neuronu (across all images)
        mean_per_neuron = act.mean(axis=0)
        std_per_neuron = act.std(axis=0)

        # Scatter: mean vs std
        ax.scatter(mean_per_neuron, std_per_neuron, alpha=0.5)
        ax.set_xlabel("Mean activation")
        ax.set_ylabel("Std activation")
        ax.set_title(f"{layer} - Neuron statistics\n({act.shape[1]} neurons)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("neuron_statistics_triangles.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Zapisano: neuron_statistics_triangles.png")


# ============================================================================
# 6. MAIN - URUCHOM WSZYSTKO
# ============================================================================

if __name__ == "__main__":
    # Ścieżka do datasetu
    DATA_DIR = "triangles_dataset"  # zmień na swoją ścieżkę

    # Sprawdź czy folder istnieje
    if not os.path.exists(DATA_DIR):
        print(f"BŁĄD: Folder {DATA_DIR} nie istnieje!")
        print("Najpierw uruchom skrypt generujący trójkąty.")
        exit()

    # Analiza
    print("=" * 70)
    print("ANALIZA SIECI NEURONOWEJ NA TRÓJKĄTACH")
    print("=" * 70)

    activations, filenames, model = analyze_network(data_dir=DATA_DIR, batch_size=16)

    # Wizualizacje
    print("\nTworzę wizualizacje...")
    visualize_activations(activations, filenames, n_samples=5)
    visualize_neuron_statistics(activations)

    print("\n" + "=" * 70)
    print("GOTOWE!")
    print("=" * 70)
    print("\nMasz teraz:")
    print("1. activations - dict z aktywnościami wszystkich warstw")
    print("2. filenames - lista nazw plików")
    print("3. model - sieć neuronowa")
    print("\nMożesz teraz:")
    print("- Analizować które neurony reagują na trójkąty")
    print("- Dodać kwadraty i porównać aktywacje")
    print("- Rozbudować sieć i powtórzyć eksperyment")
