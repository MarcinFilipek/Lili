# analyze_both.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# ============================================================================
# IMPORTY - poprawione
# ============================================================================

# Z analyze_triangles importuj tylko to co tam jest:
from analyze_triangles import SimpleConvNet, ActivationExtractor
from better_conv_net import BetterConvNet

# Nie możemy importować SquareDataset z analyze_triangles bo tam jest TriangleDataset
# Zamiast tego zdefiniujmy uniwersalną klasę tutaj:


class ShapeDataset(Dataset):
    """
    Dataset do wczytywania kształtów z folderu.
    Uniwersalna wersja - działa dla trójkątów, kwadratów, etc.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Znajdź wszystkie pliki PNG
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]
        self.image_files.sort()

        print(f"Znaleziono {len(self.image_files)} obrazów w {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]


# ============================================================================
# GŁÓWNA FUNKCJA ANALIZY
# ============================================================================


def analyze_both_datasets(triangles_dir, squares_dir, batch_size=16, seed=42):
    """
    Analizuje oba datasety używając TEGO SAMEGO modelu.

    Args:
        triangles_dir: folder z trójkątami
        squares_dir: folder z kwadratami
        batch_size: rozmiar batcha
        seed: seed dla reprodukowalności
    """

    # Ustaw seed dla reprodukowalności
    torch.manual_seed(seed)
    np.random.seed(seed)

    # JEDEN MODEL DLA OBU DATASETÓW
    model = BetterConvNet()
    model.eval()

    print("\n" + "=" * 70)
    print("ANALIZA OBU DATASETÓW Z TYM SAMYM MODELEM")
    print("=" * 70)
    print(f"\nSeed: {seed}")
    print(f"Architektura sieci:")
    print(model)
    print(f"Liczba parametrów: {sum(p.numel() for p in model.parameters()):,}")

    # Zapisz wagi dla reprodukowalności
    torch.save(model.state_dict(), f"shared_model_weights_seed{seed}.pth")
    print(f"\nZapisano wagi modelu do: shared_model_weights_seed{seed}.pth")

    # Setup transformacji
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Przygotuj oba datasety
    print("\n" + "-" * 70)
    tri_dataset = ShapeDataset(triangles_dir, transform=transform)
    sq_dataset = ShapeDataset(squares_dir, transform=transform)

    tri_loader = DataLoader(
        tri_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    sq_loader = DataLoader(
        sq_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Ekstraktor aktywacji
    extractor = ActivationExtractor(model)
    extractor.register_hooks(["conv1", "conv2", "conv3"])

    # ========================================================================
    # ANALIZA TRÓJKĄTÓW
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALIZA TRÓJKĄTÓW")
    print("=" * 70)

    tri_activations = {"conv1": [], "conv2": [], "conv3": [], "output": []}
    tri_filenames = []

    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(tri_loader):
            output = model(images)

            for layer_name in ["conv1", "conv2", "conv3"]:
                act = extractor.activations[layer_name]
                act_pooled = act.mean(dim=[2, 3])
                tri_activations[layer_name].append(act_pooled.numpy())

            tri_activations["output"].append(output.numpy())
            tri_filenames.extend(filenames)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Przetworzono {(batch_idx + 1) * batch_size} trójkątów")

    # Połącz batche
    for key in tri_activations:
        tri_activations[key] = np.vstack(tri_activations[key])

    print(f"\nZebrano aktywacje trójkątów:")
    for key, value in tri_activations.items():
        print(f"  {key}: shape {value.shape}")

    # ========================================================================
    # ANALIZA KWADRATÓW (TEN SAM MODEL!)
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALIZA KWADRATÓW (ten sam model!)")
    print("=" * 70)

    sq_activations = {"conv1": [], "conv2": [], "conv3": [], "output": []}
    sq_filenames = []

    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(sq_loader):
            output = model(images)

            for layer_name in ["conv1", "conv2", "conv3"]:
                act = extractor.activations[layer_name]
                act_pooled = act.mean(dim=[2, 3])
                sq_activations[layer_name].append(act_pooled.numpy())

            sq_activations["output"].append(output.numpy())
            sq_filenames.extend(filenames)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Przetworzono {(batch_idx + 1) * batch_size} kwadratów")

    # Połącz batche
    for key in sq_activations:
        sq_activations[key] = np.vstack(sq_activations[key])

    print(f"\nZebrano aktywacje kwadratów:")
    for key, value in sq_activations.items():
        print(f"  {key}: shape {value.shape}")

    # Cleanup
    extractor.remove_hooks()

    return tri_activations, sq_activations, tri_filenames, sq_filenames, model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Ścieżki
    TRIANGLES_DIR = "triangles_dataset"
    SQUARES_DIR = "squares_dataset"

    # Sprawdź czy istnieją
    if not os.path.exists(TRIANGLES_DIR):
        print(f"BŁĄD: Folder {TRIANGLES_DIR} nie istnieje!")
        exit()

    if not os.path.exists(SQUARES_DIR):
        print(f"BŁĄD: Folder {SQUARES_DIR} nie istnieje!")
        exit()

    # Główna analiza
    tri_act, sq_act, tri_files, sq_files, model = analyze_both_datasets(
        TRIANGLES_DIR,
        SQUARES_DIR,
        batch_size=16,
        seed=42,  # możesz zmienić żeby przetestować różne losowe inicjalizacje
    )

    # Teraz możesz użyć funkcji z compare_shapes.py
    print("\n" + "=" * 70)
    print("URUCHAMIAM ANALIZĘ PORÓWNAWCZĄ")
    print("=" * 70)

    # Import funkcji porównawczych
    from compare_shapes import (
        compute_selectivity_metrics,
        print_statistics,
        plot_comprehensive_comparison,
        plot_all_layers_summary,
        plot_pca_comparison,
    )

    # Analiza statystyczna
    print("\n" + "=" * 70)
    print("STATYSTYKI")
    print("=" * 70)

    all_layer_metrics = {}
    for layer in ["conv1", "conv2", "conv3", "output"]:
        metrics = compute_selectivity_metrics(tri_act, sq_act, layer)
        all_layer_metrics[layer] = metrics
        print_statistics(metrics, layer)

    # Wizualizacje
    print("\n" + "=" * 70)
    print("WIZUALIZACJE")
    print("=" * 70)

    print("\n[1/4] Kompleksowa analiza conv3...")
    plot_comprehensive_comparison(tri_act, sq_act, layer="conv3", save_prefix="fixed_")

    print("\n[2/4] Podsumowanie wszystkich warstw...")
    plot_all_layers_summary(tri_act, sq_act, save_prefix="fixed_")

    print("\n[3/4] PCA analysis...")
    plot_pca_comparison(tri_act, sq_act, layer="conv3", save_prefix="fixed_")

    print("\n[4/4] Analiza output layer...")
    plot_comprehensive_comparison(tri_act, sq_act, layer="output", save_prefix="fixed_")

    # Finalne podsumowanie
    print("\n" + "=" * 70)
    print("FINALNE PODSUMOWANIE")
    print("=" * 70)

    for layer in ["conv1", "conv2", "conv3", "output"]:
        m = all_layer_metrics[layer]
        print(f"\n{layer}:")
        print(f"  Korelacja: {m['correlation']:.3f}")
        print(
            f"  Neurony znaczące (p<0.05): {m['significant_neurons']}/{m['total_neurons']} ({m['significant_neurons']/m['total_neurons']*100:.1f}%)"
        )
        print(f"  Duży effect size (|d|>0.8): {(np.abs(m['cohens_d']) > 0.8).sum()}")

    print("\n" + "=" * 70)
    print("GOTOWE! Wygenerowane pliki:")
    print("=" * 70)
    print(f"  - shared_model_weights_seed42.pth")
    print(f"  - fixed_comprehensive_comparison_conv3.png")
    print(f"  - fixed_comprehensive_comparison_output.png")
    print(f"  - fixed_all_layers_summary.png")
    print(f"  - fixed_pca_comparison_conv3.png")
